from feistelStructure import FeistelStructure
import numpy as np

block_size = 32
num_rounds = 12
key_length_bytes = 16
r = 3.9

feistel = FeistelStructure(block_size, num_rounds, key_length_bytes, r)

feistel.running_time()
key = feistel.generate_key()

# feistel.encrypt_file("test files/basic.txt",
#                      "output files/encrypted files/basic.txt", key)

# feistel.decrypt_file("output files/encrypted files/basic.txt",
#                      "output files/decrypted files/basic.txt", key)

input_text = "Read the introduction and conclusion: These sections usually summarize the paper's main argument or thesis"
input_data = feistel.string_to_ascii(input_text)

avalanche_effect = feistel.avalanche_effect(input_data, key)


encrypted_data = feistel.encrypt_data(input_data, key)
encrypted_text = feistel.ascii_to_string(encrypted_data)

decrypted_data = feistel.decrypt_data(encrypted_data, key)
decrypted_text = feistel.ascii_to_string(decrypted_data)


# feistel.frequency_histogram(input_data, "input_histogram.png")
# feistel.frequency_histogram(encrypted_data, 'encrypted_histogram.png')

print("Decrypted text: ", decrypted_text)
print("Decryption matches input: ", np.array_equal(input_data, decrypted_data))
# print("Confusion score: ", feistel.confusion_score(input_block, decrypted_data))
# print("Diffusion score: ", feistel.diffusion_score(input_block, decrypted_data))
