// cargo test deepfold -- --nocapture
// ported from https://github.com/guo-yanpei/deepfold-bench
use crate::{
    pcs::{
        multilinear::{err_too_many_variates, validate_input},
        Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::PrimeField,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};

use crate::pcs::multilinear::basefold::interpolate_over_boolean_hypercube;
use crate::util::hash::{Hash, Output};
use crate::util::{
    deepfold_backend::{
        self, prover::Prover as DFProver, verifier::Verifier as DFVerifier, Commit as DFCommit,
        Proof as DFProof,
    },
    deepfold_util::{
        algebra::{
            coset::Coset, field::MyField,
            polynomial::MultilinearPolynomial as DFMultilinearPolynomial,
        },
        merkle_tree::MERKLE_ROOT_SIZE,
        random_oracle::RandomOracle,
        CODE_RATE, SECURITY_BITS, STEP,
    },
};
use rand::RngCore;
use std::{borrow::Cow, iter, marker::PhantomData, mem::size_of, slice};

#[derive(Clone, Debug)]
pub struct DeepFold<F: PrimeField, H: Hash>(PhantomData<(F, H)>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepFoldParams<F: PrimeField + MyField> {
    num_vars: usize,
    // Store oracle as a param to share oracles between commit, open, and verify
    oracle: RandomOracle<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + MyField> DeepFoldParams<F> {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
    pub fn oracle(&self) -> &RandomOracle<F> {
        &self.oracle
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct DeepFoldCommitment<F: PrimeField, H: Hash> {
    pub merkle_root: [u8; MERKLE_ROOT_SIZE],
    pub deep: F,
    pub chunks: Vec<Output<H>>,
}

impl<F: PrimeField, H: Hash> DeepFoldCommitment<F, H> {
    pub fn new() -> Self {
        Self {
            merkle_root: [0u8; MERKLE_ROOT_SIZE],
            deep: F::ZERO,
            chunks: Vec::new(),
        }
    }
}

impl<F: PrimeField, H: Hash> AsRef<[Output<H>]> for DeepFoldCommitment<F, H> {
    fn as_ref(&self) -> &[Output<H>] {
        &self.chunks
    }
}

// convert plonkish MLP to DF MLP helper
fn to_df_poly<F>(poly: &MultilinearPolynomial<F>) -> DFMultilinearPolynomial<F>
where
    F: PrimeField + MyField,
{
    let evals = poly.evals().to_vec(); // size = 2^num_vars
    let coeffs = interpolate_over_boolean_hypercube(evals);
    DFMultilinearPolynomial::new(coeffs)
}

// cosets helper
fn build_interpolate_cosets<F: MyField>(num_vars: usize) -> Vec<Coset<F>> {
    let mut cosets = vec![Coset::new(1 << (num_vars + CODE_RATE), F::from_int(1))];
    for i in 1..=num_vars {
        cosets.push(cosets[i - 1].pow(2));
    }
    cosets
}

impl<F, H> PolynomialCommitmentScheme<F> for DeepFold<F, H>
where
    F: PrimeField + MyField + Serialize + DeserializeOwned,
    H: Hash,
{
    type Param = DeepFoldParams<F>;
    type ProverParam = DeepFoldParams<F>;
    type VerifierParam = DeepFoldParams<F>;
    type Polynomial = MultilinearPolynomial<F>;
    type Commitment = DeepFoldCommitment<F, H>;
    type CommitmentChunk = Output<H>;

    fn setup(
        poly_size: usize,
        _batch_size: usize,
        _rng: impl RngCore,
    ) -> Result<Self::Param, Error> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;

        // Create single oracle and store
        let oracle = RandomOracle::<F>::new(num_vars, SECURITY_BITS / CODE_RATE);

        Ok(DeepFoldParams {
            num_vars,
            oracle,
            _marker: PhantomData,
        })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        _batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;

        if num_vars > param.num_vars() {
            return Err(err_too_many_variates("trim", param.num_vars(), num_vars));
        }

        let trimmed = DeepFoldParams {
            num_vars,
            oracle: param.oracle.clone(),
            _marker: PhantomData,
        };
        Ok((trimmed.clone(), trimmed))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.num_vars(), [poly], None)?;

        // 1) Convert plonkish poly (evals on {0,1}^n) -> DeepFold poly (coeffs)
        let df_poly = to_df_poly(poly);

        // 2) Build DeepFold cosets + oracle
        let interpolate_cosets = build_interpolate_cosets::<F>(pp.num_vars());
        let oracle = &pp.oracle;
        let total_round = pp.num_vars();

        // 3) Run DeepFold prover just to get the commitment
        let prover = DFProver::new(total_round, &interpolate_cosets, df_poly, &oracle, STEP);
        let backend_commit: DFCommit<F> = prover.commit_polynomial();

        // 4) Encode (merkle_root, deep) into chunks of Output<H>
        let mut chunks = Vec::<Output<H>>::new();

        // chunk 0: merkle_root
        let mut merkle_chunk = Output::<H>::default();
        let merkle_bytes = merkle_chunk.as_mut();
        assert!(MERKLE_ROOT_SIZE <= merkle_bytes.len());
        merkle_bytes[..MERKLE_ROOT_SIZE].copy_from_slice(&backend_commit.merkle_root);
        chunks.push(merkle_chunk);

        // chunk 1: deep as field bytes
        let mut deep_chunk = Output::<H>::default();
        let deep_chunk_bytes = deep_chunk.as_mut();
        let deep_repr = backend_commit.deep.to_repr();
        let deep_bytes = deep_repr.as_ref();
        assert!(deep_bytes.len() <= deep_chunk_bytes.len());
        deep_chunk_bytes[..deep_bytes.len()].copy_from_slice(deep_bytes);
        chunks.push(deep_chunk);

        Ok(DeepFoldCommitment {
            merkle_root: backend_commit.merkle_root,
            deep: backend_commit.deep,
            chunks,
        })
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        let polys: Vec<_> = polys.into_iter().collect();
        if polys.is_empty() {
            return Ok(Vec::new());
        }

        validate_input("batch commit", pp.num_vars(), polys.iter().copied(), None)?;

        polys
            .into_iter()
            .map(|poly| Self::commit(pp, poly))
            .collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        // 1) Standard plonkish input sanity
        validate_input("open", pp.num_vars(), [poly], [point])?;

        // 1a) Write the opening point and claimed eval into the FS transcript
        for coord in point.iter() {
            transcript.write_field_element(coord)?;
        }
        transcript.write_field_element(eval)?;

        // 2) Convert plonkish multilinear (evals) -> DeepFold multilinear (coeffs)
        let df_poly = to_df_poly(poly);

        // 3) Build cosets + random oracle exactly like in commit
        let interpolate_cosets = build_interpolate_cosets::<F>(pp.num_vars());
        let oracle = &pp.oracle;

        // 4) Run DeepFold prover, get backend commitment
        let mut prover =
            DFProver::<F>::new(pp.num_vars(), &interpolate_cosets, df_poly, &oracle, STEP);
        let backend_commit: DFCommit<F> = prover.commit_polynomial();

        debug_assert_eq!(backend_commit.merkle_root, comm.merkle_root);
        debug_assert_eq!(backend_commit.deep, comm.deep);

        // 6) Build DeepFold verifier from the backend commit
        let mut df_verifier = DFVerifier::<F>::new(
            pp.num_vars(),
            &interpolate_cosets,
            backend_commit,
            &oracle,
            STEP,
        );

        // 7) Convert Point<F, Polynomial> -> Vec<F> for DeepFold
        let open_point: Vec<F> = point.clone();
        df_verifier.set_open_point(&open_point);

        // 8) Generate DeepFold proof at this point
        let backend_proof = prover.generate_proof(open_point);

        // 9) Check that DeepFold’s verifier accepts the proof
        debug_assert!(df_verifier.verify(backend_proof.clone()));
        debug_assert_eq!(poly.evaluate(point), *eval);

        // 10) Serialize the DeepFold proof and push it into the transcript
        use bincode;

        let proof_bytes = bincode::serialize(&backend_proof)
            .expect("DeepFold proof serialization should not fail");

        // write the length as a commitment chunk
        let proof_len = proof_bytes.len() as u64;
        let mut len_chunk = Output::<H>::default();
        let len_bytes = proof_len.to_le_bytes();
        len_chunk.as_mut()[..8].copy_from_slice(&len_bytes);
        transcript.write_commitment(&len_chunk)?;

        // write the proof bytes as chunks
        let chunk_len = Output::<H>::default().as_ref().len();
        for chunk in proof_bytes.chunks(chunk_len) {
            let mut out = Output::<H>::default();
            out.as_mut()[..chunk.len()].copy_from_slice(chunk);
            transcript.write_commitment(&out)?;
        }
        Ok(())
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        let polys: Vec<_> = polys.into_iter().collect();
        let comms: Vec<_> = comms.into_iter().collect();

        assert_eq!(polys.len(), comms.len());
        assert_eq!(polys.len(), evals.len());

        for (((poly, comm), point), eval) in polys
            .iter()
            .zip(comms.iter())
            .zip(points.iter())
            .zip(evals.iter())
        {
            Self::open(pp, poly, comm, point, &eval.value, transcript)?;
        }

        Ok(())
    }

    fn read_commitments(
        _vp: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        let chunk_len = Output::<H>::default().as_ref().len();
        let fe_repr_len = <F as PrimeField>::Repr::default().as_ref().len();

        if MERKLE_ROOT_SIZE > chunk_len || fe_repr_len > chunk_len {
            return Err(Error::InvalidPcsOpen(
                "DeepFold commitment chunk size too small".to_string(),
            ));
        }

        let mut comms = Vec::with_capacity(num_polys);

        for _ in 0..num_polys {
            // read chunk 0: merkle_root
            let chunk0: Output<H> = transcript.read_commitment()?;
            let bytes0 = chunk0.as_ref();
            let mut merkle_root = [0u8; MERKLE_ROOT_SIZE];
            merkle_root.copy_from_slice(&bytes0[..MERKLE_ROOT_SIZE]);

            // read chunk 1: deep
            let chunk1: Output<H> = transcript.read_commitment()?;
            let bytes1 = chunk1.as_ref();
            let mut repr = <F as PrimeField>::Repr::default();
            repr.as_mut().copy_from_slice(&bytes1[..fe_repr_len]);
            let deep = F::from_repr_vartime(repr).ok_or_else(|| {
                Error::InvalidPcsOpen("Invalid DeepFold commitment encoding".to_string())
            })?;

            comms.push(DeepFoldCommitment {
                merkle_root,
                deep,
                chunks: vec![chunk0, chunk1],
            });
        }

        Ok(comms)
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        // 1) Read point coordinates back
        let num_vars = vp.num_vars();
        let transcript_point = transcript.read_field_elements(num_vars)?;
        let transcript_eval = transcript.read_field_element()?;

        if transcript_point != *point || transcript_eval != *eval {
            return Err(Error::InvalidPcsOpen("Proof error".to_string()));
        }

        // 2) Read proof length as a commitment chunk
        let len_chunk: Output<H> = transcript.read_commitment()?;
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&len_chunk.as_ref()[..8]);
        let proof_len = u64::from_le_bytes(len_bytes) as usize;

        // 3) Read proof bytes
        let chunk_len = Output::<H>::default().as_ref().len();
        let num_chunks = (proof_len + chunk_len - 1) / chunk_len;

        let mut proof_bytes = Vec::with_capacity(num_chunks * chunk_len);
        for _ in 0..num_chunks {
            let chunk: Output<H> = transcript.read_commitment()?;
            proof_bytes.extend_from_slice(chunk.as_ref());
        }
        proof_bytes.truncate(proof_len);

        // 4) Deserialize the backend proof
        let backend_proof: DFProof<F> = bincode::deserialize(&proof_bytes)
            .map_err(|_| Error::InvalidPcsOpen("Proof error".to_string()))?;

        // 5) Rebuild cosets + oracle + backend verifier, then run it
        let interpolate_cosets = build_interpolate_cosets::<F>(vp.num_vars());
        let oracle = &vp.oracle;

        let backend_commit = DFCommit {
            merkle_root: comm.merkle_root,
            deep: comm.deep,
        };

        let mut df_verifier = DFVerifier::<F>::new(
            vp.num_vars(),
            &interpolate_cosets,
            backend_commit,
            &oracle,
            STEP,
        );

        df_verifier.set_open_point(&point.to_vec());

        if !df_verifier.verify(backend_proof) {
            return Err(Error::InvalidPcsOpen("Proof error".to_string()));
        }

        Ok(())
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        let comms: Vec<_> = comms.into_iter().collect();
        assert_eq!(comms.len(), evals.len());

        for ((comm, point), eval) in comms.iter().zip(points.iter()).zip(evals.iter()) {
            Self::verify(vp, comm, point, &eval.value, transcript)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::DeepFold;
    use crate::{
        pcs::multilinear::test::{run_batch_commit_open_verify, run_commit_open_verify},
        util::transcript::Blake2s256Transcript,
    };
    use blake2::{digest::FixedOutputReset, Blake2s256};
    use halo2_curves::bn256::Fr;

    type Pcs = DeepFold<Fr, Blake2s256>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Blake2s256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Blake2s256Transcript<_>>();
    }
}
