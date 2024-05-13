from feistelStructure import FeistelStructure
import numpy as np

block_size = 1024
num_rounds = 16
key_length_bytes = 512
r = 3.9

feistel = FeistelStructure(block_size, num_rounds, key_length_bytes, r)

# feistel.running_time()
key = feistel.generate_key()

# feistel.encrypt_file("test files/basic.txt",
#                      "output files/encrypted files/basic.txt", key)

# feistel.decrypt_file("output files/encrypted files/basic.txt",
#                      "output files/decrypted files/basic.txt", key)

input_text2 = "Read the introduction and conclusion: These sections usually summarize the paper's main argument or thesis"
input_text = '''To observe its chaotic behaviors, its bifurcation diagram and Lyapunov Exponent are presented in
Fig. 2(a) and Fig. 3(a). In the bifurcation diagram shown in Fig. 2(a), the dotted line shows its good chaotic
behavior and the solid line represents its non-chaotic property. There are two problems in the Logistic map.
First, its chaotic range is limited only within [3.57, 4]. Even within this range, there are some parameters
which make the Logistic map to have no chaotic behaviors. This is verified by the blank zone in its bi-
furcation diagram and plot of the Lyapunov Exponent in Fig. 3(a). For the Lyapunov Exponent, a positive
value means a good chaotic property of a chaotic map. As shown in Fig. 3(a), the Lyapunov Exponents
of the Logistic map are smaller than zero when parameter r < 3.57. Second, the data range of the chaotic'''

input_data = feistel.string_to_ascii(input_text)

avalanche_effect = feistel.avalanche_effect(input_data, key)
print("Avalanche effect: ", avalanche_effect)


encrypted_data = feistel.encrypt_data(input_data, key)
encrypted_text = feistel.ascii_to_string(encrypted_data)
# print("Encrypted Text: ", encrypted_text)

decrypted_data = feistel.decrypt_data(encrypted_data, key)
decrypted_text = feistel.ascii_to_string(decrypted_data)


# feistel.frequency_histogram(input_data, "input_histogram.png")
# feistel.frequency_histogram(encrypted_data, 'encrypted_histogram.png')

# print("Decrypted text: ", decrypted_text)
print("Decryption matches input: ", np.array_equal(input_data, decrypted_data))
# print("Confusion score: ", feistel.confusion_score(input_block, decrypted_data))
# print("Diffusion score: ", feistel.diffusion_score(input_block, decrypted_data))

# feistel.encrypt_file("test files/basic.txt",
#                      "output files/encrypted files/basic.txt", key)
# feistel.decrypt_file("output files/encrypted files/basic.txt",
#                      "output files/decrypted files/basic.txt", key)
