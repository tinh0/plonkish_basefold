// MyField implementation for halo2 for deepfold codes

use super::MyField;
use crate::util::deepfold_util::merkle_tree::MERKLE_ROOT_SIZE;
use ff::{Field, PrimeField};
use halo2_curves::bn256::Fr;
impl MyField for Fr {
    const FIELD_NAME: &'static str = "bn256::Fr";
    const LOG_ORDER: u64 = 28;

    fn from_int(x: u64) -> Self {
        Fr::from(x)
    }

    fn random_element() -> Self {
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        let mut buf = [0u8; 32];
        rng.fill_bytes(&mut buf);
        // map 32 bytes into the field
        let ct = Fr::from_bytes(&buf);
        Option::<Fr>::from(ct).unwrap_or_else(Fr::zero)
    }

    fn inverse(&self) -> Self {
        // ff::Field::invert returns CtOption<Self>
        Option::<Fr>::from(self.invert()).unwrap_or_else(Fr::zero)
    }

    fn is_zero(&self) -> bool {
        // avoid ambiguity with MyField::is_zero by using fully-qualified call
        <Fr as Field>::is_zero(self).into()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_repr().as_ref().to_vec()
    }

    fn from_hash(hash: [u8; MERKLE_ROOT_SIZE]) -> Self {
        // interpret first 32 bytes of the hash as a field element
        let mut buf = [0u8; 32];
        let take = core::cmp::min(MERKLE_ROOT_SIZE, 32);
        buf[..take].copy_from_slice(&hash[..take]);
        let ct = Fr::from_bytes(&buf);
        Option::<Fr>::from(ct).unwrap_or_else(Fr::zero)
    }

    fn root_of_unity() -> Self {
        // Use halo2 BN256 primitive root of unity
        // This constant is provided by the field_common! macro internally as ROOT_OF_UNITY
        Fr::ROOT_OF_UNITY
    }

    fn inverse_2() -> Self {
        let two = Fr::from(2u64);
        Option::<Fr>::from(two.invert()).unwrap_or_else(Fr::zero)
    }
}
