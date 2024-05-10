## Fiestel Cipher Implementation

This repository contains a Python implementation of a Fiestel cipher structure. The Fiestel cipher is a symmetric key encryption algorithm that operates on blocks of data. It uses a structure comprising multiple rounds of processing to achieve encryption and decryption.

### Architecture Overview

The Fiestel cipher structure consists of the following components:

- **FiestelStructure Class**: This class represents the Fiestel cipher structure and provides methods for encryption, decryption, key generation, and performance evaluation.

- **Chaotic Map Function**: The chaotic_map function computes the next value in a chaotic map sequence. It is used within the Fiestel cipher rounds to introduce randomness.

- **Key Generation**: The generate_key method generates a random key of specified length.

- **Encryption and Decryption**: The encrypt_block and decrypt_block methods perform encryption and decryption for a single block of data. The encrypt_data and decrypt_data methods handle encryption and decryption for sequences of data blocks.

- **Performance Evaluation**: The confusion_score and diffusion_score methods compute metrics to evaluate the effectiveness of the encryption process.

- **String Conversion**: The string_to_bits and bits_to_string methods convert between strings and their binary representations.

### Usage

To use the Fiestel cipher implementation:

1. **Instantiate FiestelStructure**: Create an instance of the FiestelStructure class, specifying parameters such as block size, number of rounds, key length, and a parameter for the chaotic map.

```python
block_size = 128
num_rounds = 16
key_length_bytes = 8
r = 3.9

fiestel = FiestelStructure(block_size, num_rounds, key_length_bytes, r)
```

2. **Generate Key**: Generate a random key using the generate_key method.

```python
key = fiestel.generate_key()
```

3. **Encrypt and Decrypt Data**: Use the encrypt_data and decrypt_data methods to encrypt and decrypt data blocks.

```python
input_text = "Your input text here"
input_block = fiestel.string_to_bits(input_text)

encrypted_data = fiestel.encrypt_data(input_block, key)
decrypted_data = fiestel.decrypt_data(encrypted_data, key)
```

4. **Evaluate Performance and Security**: Optionally, evaluate the performance and security of the encryption process using the confusion_score and diffusion_score methods.

```python
confusion = fiestel.confusion_score(input_block, decrypted_data)
diffusion = fiestel.diffusion_score(input_block, decrypted_data)
```

### Running Time Evaluation

The running_time method can be used to evaluate the running time of the encryption and decryption processes. By default, it reads test files from a directory and performs encryption and decryption on a specified amount of data.

```python
fiestel.running_time(data_size=105000)
```

### Example

An example usage demonstrating encryption and decryption of text data is provided below:

```python
input_text = "Read the introduction and conclusion: These sections usually summarize the paper's main argument or thesis."
input_block = fiestel.string_to_bits(input_text)

encrypted_data = fiestel.encrypt_data(input_block, key)
decrypted_data = fiestel.decrypt_data(encrypted_data, key)
decrypted_text = fiestel.bits_to_string(decrypted_data)

print("Decrypted text: ", decrypted_text)
print("Decryption matches input: ", np.array_equal(input_block, decrypted_data))
```

### Dependencies

- numpy
- secrets
- os
- timeit

### Notes

- Ensure that the data size and key length are appropriate for your application to achieve desired security levels and performance.

### Contributors

- [Your Name or Organization](link_to_github_profile)

### License

[License Name] (link_to_license_file)
