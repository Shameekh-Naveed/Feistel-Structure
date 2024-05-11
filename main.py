from fiestelStructure import FiestelStructure


block_size = 128
num_rounds = 2
key_length_bytes = 64
r = 3.9

fiestel = FiestelStructure(block_size, num_rounds, key_length_bytes, r)

# fiestel.running_time()
key = fiestel.generate_key()
print("Key: ", key)

# fiestel.encrypt_file("test files/basic.txt",
#                      "output files/encrypted files/basic.txt", key)

# fiestel.decrypt_file("output files/encrypted files/basic.txt",
#                      "output files/decrypted files/basic.txt", key)

# input_text = "Read the introduction and conclusion: These sections usually summarize the paper's main argument or thesis."
# input_data = fiestel.string_to_ascii(input_text)

# encrypted_data = fiestel.encrypt_data(input_data, key)
# encrypted_text = fiestel.ascii_to_string(encrypted_data)
# print("encrypted_text", encrypted_text)

# decrypted_data = fiestel.decrypt_data(encrypted_data, key)
# decrypted_text = fiestel.ascii_to_string(decrypted_data)


# fiestel.frequency_histogram(input_data, "input_histogram.png")
# fiestel.frequency_histogram(encrypted_data, 'encrypted_histogram.png')

# print("Decrypted text: ", decrypted_text)
# print("Decryption matches input: ", np.array_equal(input_data, decrypted_data))
# print("Confusion score: ", fiestel.confusion_score(input_block, decrypted_data))
# print("Diffusion score: ", fiestel.diffusion_score(input_block, decrypted_data))
