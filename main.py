import numpy as np
import secrets


class FiestelStructure:
    def __init__(self, block_size, num_rounds, key_length_bytes, r):
        self.block_size = block_size
        self.num_rounds = num_rounds
        self.key_length_bytes = key_length_bytes
        self.r = r

    def chaotic_map(self, x):
        return self.r * x * (1 - x)

    def generate_key(self):
        key = secrets.token_bytes(self.key_length_bytes)
        key_bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8))
        return key_bits

    def feistel_round(self, L, R, round_key):
        C = self.chaotic_map(R)
        C_int = C.astype(int)
        F = L ^ C_int
        F ^= round_key
        return R, F

    def encrypt_block(self, block, key):
        L, R = np.split(block, 2)
        round_keys = [key] * self.num_rounds
        for i in range(self.num_rounds):
            L, R = self.feistel_round(L, R, round_keys[i])
        encrypted_block = np.concatenate((R, L))
        return encrypted_block

    def encrypt_data(self, data, key):
        encrypted_data = []
        for i in range(0, len(data), self.block_size):
            block = data[i:i+self.block_size]
            if len(block) < self.block_size:
                block = np.concatenate(
                    (block, np.zeros(self.block_size - len(block), dtype=int)))
            encrypted_block = self.encrypt_block(block, key)
            encrypted_data.append(encrypted_block)
        return np.concatenate(encrypted_data)

    def decrypt_block(self, block, key):
        L, R = np.split(block, 2)
        round_keys = [key] * self.num_rounds
        round_keys = round_keys[::-1]
        for i in range(self.num_rounds):
            L, R = self.feistel_round(L, R, round_keys[i])
        decrypted_block = np.concatenate((R, L))
        return decrypted_block

    def decrypt_data(self, encrypted_data, key):
        decrypted_data = []
        for i in range(0, len(encrypted_data), self.block_size):
            block = encrypted_data[i:i+self.block_size]
            decrypted_block = self.decrypt_block(block, key)
            decrypted_data.append(decrypted_block)
        return np.concatenate(decrypted_data)

    def confusion_score(self, plaintext, ciphertext):
        unique_ct = np.unique(ciphertext)
        prob_ct = np.array(
            [np.count_nonzero(ciphertext == ct) / len(ciphertext)
             for ct in unique_ct]
        )
        confusion = np.sum(prob_ct ** 2)
        return confusion

    def diffusion_score(self, plaintext, ciphertext):
        hamming_dist = np.sum(plaintext != ciphertext)
        hamming_dist /= len(plaintext)
        return hamming_dist

    def string_to_bits(self, string):
        return np.array([int(i) for i in "".join(format(ord(i), "08b") for i in string)])

    def bits_to_string(self, bits):
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
print("Encrypted data: ", encrypted_data.shape)
print("Input block: ", input_block.shape)
decrypted_data = fiestel.decrypt_data(encrypted_data, key)
decrypted_text = fiestel.bits_to_string(decrypted_data)

print("Decrypted text: ", decrypted_text)
print("Decryption matches input: ", np.array_equal(input_block, decrypted_data))
print("Confusion score: ", fiestel.confusion_score(input_block, decrypted_data))
print("Diffusion score: ", fiestel.diffusion_score(input_block, decrypted_data))
