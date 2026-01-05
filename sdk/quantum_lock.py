import numpy as np
import pickle
import base64

class QuantumLock:
    """
    QuantumLock: A Post-Quantum Cryptography SDK.
    Algorithm: Learning With Errors (LWE) - Simplified Lattice Cryto.
    
    Concept:
    Recovering a vector 's' from 'As + e' is hard for Quantum Computers,
    akin to untangling a knot in high-dimensional space without knowing
    the error 'e'.
    """
    
    def __init__(self, n=512, q=3329, sigma=1.0):
        self.n = n        # Lattice Dimension (Security Level)
        self.q = q        # Modulus
        self.sigma = sigma # Noise Standard Deviation
        self.pk = None
        self.sk = None
        
    def box_muller(self, shape):
        # Generate Gaussian noise errors
        u1 = np.random.rand(*shape)
        u2 = np.random.rand(*shape)
        z = np.sqrt(-2.0 * np.log(u1 + 1e-9)) * np.cos(2 * np.pi * u2)
        return np.round(z * self.sigma).astype(int)

    def keygen(self):
        """Generates Public (A, b) and Private (s) keys."""
        print("Generating Lattice Keys...")
        # 1. Private Key (s): Random vector
        s = np.random.randint(0, self.q, self.n)
        
        # 2. Public Matrix (A): Large random matrix
        # For efficiency in this demo, we use n x n, usually A is m x n
        A = np.random.randint(0, self.q, (self.n, self.n))
        
        # 3. Error (e): Gaussian noise
        e = self.box_muller((self.n,))
        
        # 4. Public Vector (b = As + e mod q)
        b = (np.dot(A, s) + e) % self.q
        
        self.sk = s
        self.pk = (A, b)
        print("Keys Generated.")
        return self.pk, self.sk

    def encrypt(self, message_bytes, pk=None):
        """
        Encrypts a byte-string message (multi-bit LWE).
        """
        if pk is None: pk = self.pk
        A, b = pk
        
        # Convert message to binary bits
        # This is a naive encoding: 1 bit per lattice sample.
        # Not efficient, but robust for demo.
        
        bits = []
        for byte in message_bytes:
            for i in range(8):
                bits.append((byte >> i) & 1)
        
        cipher_u = []
        cipher_v = []
        
        for bit in bits:
            # For each bit, we choose a small random vector r (Sample)
            # r should be sparse/small
            r = np.random.randint(0, 2, self.n) # 0 or 1
            
            # Encrypt bit 'm'
            # u = A^T r
            # v = b^T r + m * (q//2)
            # All operations mod q
            
            u = np.dot(A.T, r) % self.q
            
            msg_scaled = bit * (self.q // 2)
            v_val = (np.dot(b.T, r) + msg_scaled) % self.q
            
            cipher_u.append(u)
            cipher_v.append(v_val)
            
        return (cipher_u, cipher_v)

    def decrypt(self, ciphertext, sk=None):
        """
        Untangles the noisy lattice point to recover the message.
        """
        if sk is None: sk = self.sk
        cipher_u, cipher_v = ciphertext
        
        decrypted_bits = []
        
        for u, v in zip(cipher_u, cipher_v):
            # Compute dec = v - s^T u
            # dec = (b^T r + m q/2) - s^T (A^T r)
            #     = (s^T A^T + e^T) r + m q/2 - s^T A^T r
            #     = e^T r + m q/2
            # e^T r is small noise. m q/2 is 0 or q/2.
            
            dec_val = (v - np.dot(sk, u)) % self.q
            
            # If closer to q/2 -> 1, if closer to 0 -> 0 (Centering)
            # Distance to 0 vs Distance to q/2
            
            # Shift to centered modulation
            if dec_val > self.q // 2:
                dec_val -= self.q
            
            # Threshold
            if abs(dec_val) > self.q // 4:
                decrypted_bits.append(1)
            else:
                decrypted_bits.append(0)
                
        # Bits to bytes
        message_out = bytearray()
        for i in range(0, len(decrypted_bits), 8):
            byte_val = 0
            for b in range(8):
                if i+b < len(decrypted_bits):
                    byte_val |= (decrypted_bits[i+b] << b)
            message_out.append(byte_val)
            
        return bytes(message_out)

if __name__ == "__main__":
    # Test
    ql = QuantumLock()
    ql.keygen()
    msg = b"Hello Quantum World"
    print(f"Original: {msg}")
    enc = ql.encrypt(msg)
    print("Encrypted (Ciphertext chunks generated)")
    dec = ql.decrypt(enc)
    print(f"Decrypted: {dec}")
