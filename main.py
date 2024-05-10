import numpy as np
import secrets


class FiestelStructure:
    """Class implementing a Fiestel cipher structure."""

    def __init__(self, block_size, num_rounds, key_length_bytes, r):
        """Initialize the Fiestel cipher structure.

        Args:
            block_size (int): The size of each block in bits.
            num_rounds (int): The number of rounds in the Feistel structure.
            key_length_bytes (int): The length of the key in bytes.
            r (float): A parameter for the chaotic map.

        Returns:
            None
        """
        self.block_size = block_size
        self.num_rounds = num_rounds
        self.key_length_bytes = key_length_bytes
        self.r = r

    def chaotic_map(self, x):
        """Compute the next value in a chaotic map sequence.

        Args:
            x (float): The current value in the sequence.

        Returns:
            float: The next value in the chaotic map sequence.
        """
        return self.r * x * (1 - x)

    def generate_key(self):
        """Generate a random key.

        Returns:
            numpy.ndarray: An array representing the generated key.
        """
        key = secrets.token_bytes(self.key_length_bytes)
        key_bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8))
        return key_bits

    def feistel_round(self, L, R, round_key):
        """Perform one round of the Feistel cipher.

        Args:
            L (numpy.ndarray): The left half of the block.
            R (numpy.ndarray): The right half of the block.
            round_key (numpy.ndarray): The key for the current round.

        Returns:
            tuple: The updated left and right halves of the block.
        """
        C = self.chaotic_map(R)
        C_int = C.astype(int)
        F = L ^ C_int
        F ^= round_key
        return R, F

    def encrypt_block(self, block, key):
        """Encrypt a single block of data.

        Args:
            block (numpy.ndarray): The block of data to be encrypted.
            key (numpy.ndarray): The encryption key.

        Returns:
            numpy.ndarray: The encrypted block.
        """
        L, R = np.split(block, 2)
        round_keys = [key] * self.num_rounds
        for i in range(self.num_rounds):
            L, R = self.feistel_round(L, R, round_keys[i])
        encrypted_block = np.concatenate((R, L))
        return encrypted_block

    def encrypt_data(self, data, key):
        """Encrypt a sequence of data blocks.

        Args:
            data (numpy.ndarray): The data to be encrypted.
            key (numpy.ndarray): The encryption key.

        Returns:
            numpy.ndarray: The encrypted data.
        """
        encrypted_data = []
        for i in range(0, len(data), self.block_size):
            block = data[i:i+self.block_size]
            if len(block) < self.block_size:
                diff = self.block_size - len(block)
                # Represent the difference in 8 bits
                diff_bits = np.unpackbits(np.array([diff], dtype=np.uint8))
                print("Diff bits: ", diff_bits)
                padding_bits = np.concatenate(
                    (np.zeros(diff - 8, dtype=np.uint8), diff_bits))
                block = np.concatenate((block, padding_bits))

            encrypted_block = self.encrypt_block(block, key)
            encrypted_data.append(encrypted_block)
        return np.concatenate(encrypted_data)

    def decrypt_block(self, block, key):
        """Decrypt a single block of data.

        Args:
            block (numpy.ndarray): The block of data to be decrypted.
            key (numpy.ndarray): The decryption key.

        Returns:
            numpy.ndarray: The decrypted block.
        """
        L, R = np.split(block, 2)
        round_keys = [key] * self.num_rounds
        round_keys = round_keys[::-1]
        for i in range(self.num_rounds):
            L, R = self.feistel_round(L, R, round_keys[i])
        decrypted_block = np.concatenate((R, L))
        return decrypted_block

    def decrypt_data(self, encrypted_data, key):
        """Decrypt a sequence of encrypted data blocks.

        Args:
            encrypted_data (numpy.ndarray): The encrypted data.
            key (numpy.ndarray): The decryption key.

        Returns:
            numpy.ndarray: The decrypted data.
        """
        decrypted_data = []
        for i in range(0, len(encrypted_data), self.block_size):
            block = encrypted_data[i:i+self.block_size]
            decrypted_block = self.decrypt_block(block, key)
            decrypted_data.append(decrypted_block)
        # Remove padding
        padding_bits = decrypted_data[-1][-8:]
        padding_size = np.packbits(padding_bits)[0]
        decrypted_data[-1] = decrypted_data[-1][:
                                                len(decrypted_data[-1]) - padding_size]

        return np.concatenate(decrypted_data)

    def confusion_score(self, plaintext, ciphertext):
        """Calculate the confusion score between plaintext and ciphertext.

        Args:
            plaintext (numpy.ndarray): The original plaintext data.
            ciphertext (numpy.ndarray): The encrypted ciphertext data.

        Returns:
            float: The confusion score.
        """
        unique_ct = np.unique(ciphertext)
        prob_ct = np.array(
            [np.count_nonzero(ciphertext == ct) / len(ciphertext)
             for ct in unique_ct]
        )
        confusion = np.sum(prob_ct ** 2)
        return confusion

    def diffusion_score(self, plaintext, ciphertext):
        """Calculate the diffusion score between plaintext and ciphertext.

        Args:
            plaintext (numpy.ndarray): The original plaintext data.
            ciphertext (numpy.ndarray): The encrypted ciphertext data.

        Returns:
            float: The diffusion score.
        """
        hamming_dist = np.sum(plaintext != ciphertext)
        hamming_dist /= len(plaintext)
        return hamming_dist

    def string_to_bits(self, string):
        """Convert a string to a binary array.

        Args:
            string (str): The input string.

        Returns:
            numpy.ndarray: The binary representation of the string.
        """
        return np.array([int(i) for i in "".join(format(ord(i), "08b") for i in string)])

    def bits_to_string(self, bits):
        """Convert a binary array to a string.

        Args:
            bits (numpy.ndarray): The input binary array.

        Returns:
            str: The decoded string.
        """
        return "".join(
            chr(int("".join(map(str, bits[i: i + 8])), 2)) for i in range(0, len(bits), 8)
        )


block_size = 128
num_rounds = 16
key_length_bytes = 8
r = 3.9

fiestel = FiestelStructure(block_size, num_rounds, key_length_bytes, r)
key = fiestel.generate_key()

input_text = "Read the introduction and conclusion: These sections usually summarize the paper's main argument or thesis."
input_block = fiestel.string_to_bits(input_text)

encrypted_data = fiestel.encrypt_data(input_block, key)
decrypted_data = fiestel.decrypt_data(encrypted_data, key)
decrypted_text = fiestel.bits_to_string(decrypted_data)

print("Decrypted text: ", decrypted_text)
print("Decryption matches input: ", np.array_equal(input_block, decrypted_data))
print("Confusion score: ", fiestel.confusion_score(input_block, decrypted_data))
print("Diffusion score: ", fiestel.diffusion_score(input_block, decrypted_data))
