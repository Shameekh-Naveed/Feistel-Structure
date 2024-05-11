import numpy as np
import secrets
import os
import timeit
import matplotlib.pyplot as plt


class FeistelStructure:
    """Class implementing a Feistel cipher structure."""

    def __init__(self, block_size, num_rounds, key_length_bytes, r):
        """Initialize the Feistel cipher structure.

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

    def logistic_chaotic_map(self, x):
        """Compute the next value in a chaotic map sequence.

        Args:
            x (float): The current value in the sequence.

        Returns:
            float: The next value in the chaotic map sequence.
        """
        return self.r * x * (1 - x)

    def sine_chaotic_map(self, x, a):
        """
        Compute the next value in a sine chaotic map sequence.

        Args:
            x (float): The current value in the sequence.
            a (float): The parameter for the chaotic map.

        Returns:
            float: The next value in the sequence.
        """
        pi = np.pi
        return a * np.sin(pi * x)

    def generate_key(self):
        """Generate a random key.

        Returns:
            numpy.ndarray: An array representing the generated key.
        """
        key = np.array([secrets.randbits(8)
                        for _ in range(self.key_length_bytes)])
        return key

    def key_expansion(self, initial_key, n):
        """
        Key expansion algorithm using a 1D logistic chaotic map.

        Args:
        initial_key: Initial key (list or array of numbers between 0 and 255 of size key_length_bytes).
        n: Number of new keys to generate.
        r: Parameter controlling the behavior of the logistic map (typically in the range 0 to 4).

        Returns:
        List of n new keys, each of the same size as the initial key.
        """
        key_size = len(initial_key)
        expanded_keys = []

        # Generate chaotic values using the logistic map
        chaotic_values = [self.logistic_chaotic_map(initial_key[0])]
        for _ in range(1, key_size * n):
            chaotic_values.append(self.logistic_chaotic_map(chaotic_values[-1]))

        # Divide chaotic values into n segments and use them to expand the initial key
        for i in range(n):
            expanded_key = [
                (initial_key[j] + chaotic_values[i*key_size + j]) % 1 for j in range(key_size)]
            expanded_keys.append(expanded_key)

        return expanded_keys

    def feistel_round(self, L, R, round_key):
        """Perform one round of the Feistel cipher.

        Args:
            L (numpy.ndarray): The left half of the block.
            R (numpy.ndarray): The right half of the block.
            round_key (numpy.ndarray): The key for the current round.

        Returns:
            tuple: The updated left and right halves of the block.
        """
        logistic_C = self.logistic_chaotic_map(R)

        sine_C = self.sine_chaotic_map(R, 2)

        C = logistic_C + sine_C
        C = C.astype(int)

        C = C % 1

        F = L ^ C
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
                padding_digits = np.array([secrets.randbelow(256)
                                           for _ in range(diff)], dtype=np.uint8)
                padding_digits[-1] = diff
                block = np.concatenate((block, padding_digits))

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
            print("Block Size: ", len(block))
            decrypted_block = self.decrypt_block(block, key)
            decrypted_data.append(decrypted_block)
        # Remove padding
        padding_size = decrypted_data[-1][-1]
        # padding_size = np.packbits(padding_bits)[0]
        decrypted_data[-1] = decrypted_data[-1][:
                                                len(decrypted_data[-1]) - padding_size]

        return np.concatenate(decrypted_data)

    def encrypt_file(self, input_file, output_file, key):
        """Encrypt a file and save the encrypted data to another file.

        Args:
            input_file (str): The path to the input file.
            output_file (str): The path to the output file.
            key (numpy.ndarray): The encryption key.

        Returns:
            None
        """
        with open(input_file, "rb") as f:
            input_data = np.frombuffer(f.read(), dtype=np.uint8)
        encrypted_data = self.encrypt_data(input_data, key)
        encrypted_data = self.ascii_to_string(encrypted_data)
        with open(output_file, "w") as f:
            f.write(encrypted_data)

    def decrypt_file(self, input_file, output_file, key):
        """Decrypt a file and save the decrypted data to another file.

        Args:
            input_file (str): The path to the input file.
            output_file (str): The path to the output file.
            key (numpy.ndarray): The decryption key.

        Returns:
            None
        """
        with open(input_file, "r") as f:
            input_data = f.read()
        # encrypted_data = np.array([char for char in input_data])
        # print("Encrypted Data: ", encrypted_data)
        # Convert the string back to a numpy array of ASCII values
        encrypted_data = self.string_to_ascii(input_data)
        print("Encrypted Data: ", encrypted_data)

        decrypted_data = self.decrypt_data(encrypted_data, key)
        decrypted_data = self.ascii_to_string(decrypted_data)
        with open(output_file, "w") as f:
            f.write(decrypted_data)

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

    def string_to_ascii(self, input_string):
        """
        Replace each character in the input string with its ASCII value and return as a numpy array.

        Args:
            input_string (str): The input string.

        Returns:
            numpy.ndarray: The array containing ASCII values of characters.
        """
        # Create a numpy array to store ASCII values
        ascii_values = np.array([ord(char) for char in input_string])
        return ascii_values

    def ascii_to_string(self, ascii_array):
        """
        Convert a numpy array of ASCII values back into a string.

        Args:
            ascii_array (numpy.ndarray): The array containing ASCII values of characters.

        Returns:
            str: The string reconstructed from ASCII values.
        """
        # Convert each ASCII value to its corresponding character and join them into a string
        string = ''.join(chr(int(value)) for value in ascii_array)

        return string

    def running_time(self, data_size=105000):
        """Calculate the running time of the encryption and decryption processes.

        Args:
            data_size (int): The size of the data in bytes. Default 105000 bytes = 105kB

        Returns:
            None : The running time of the encryption and decryption processes are printed.
        """
        # Fetch all the files in a directory
        directory = "test files/"
        files = os.listdir(directory)
        for file in files:
            with open(directory + file, "rb") as f:
                input_data = f.read(data_size)
                # Imported Files are already in ASCII
                input_data = np.frombuffer(input_data, dtype=np.uint8)
                key = self.generate_key()
                encrypted_data = self.encrypt_data(input_data, key)
                decrypted_data = self.decrypt_data(encrypted_data, key)

                self.frequency_histogram(input_data, "input_histogram.png")
                self.frequency_histogram(
                    encrypted_data, 'encrypted_histogram.png')

                print("File: ", file)
                print("Encryption time: ", timeit.timeit(
                    lambda: self.encrypt_data(input_data, key), number=1), "s")
                print("Decryption time: ", timeit.timeit(
                    lambda: self.decrypt_data(encrypted_data, key), number=1), "s")
                print("Decryption matches input: ",
                      np.array_equal(input_data, decrypted_data))
                print("\n")
                break

    def frequency_histogram(self, input_numbers, output_file='histogram.png'):
        """
        Generate a histogram of number frequencies from a given array of numbers and save it as a PNG file.

        Args:
            input_numbers (list or array-like): The input array of numbers.
            output_file (str): The filename to save the histogram as a PNG file. Default is 'histogram.png'.

        Returns:
            None
        """
        # Count occurrences of each number in the input array
        number_counts = {number: np.count_nonzero(
            input_numbers == number) for number in set(input_numbers)}

        # Sort numbers
        sorted_numbers = sorted(number_counts.keys())

        # Print the maximum and minimum numbers
        print("Minimum number: ", min(sorted_numbers))
        print("Maximum number: ", max(sorted_numbers))

        # Extract numbers and corresponding counts
        numbers = sorted_numbers
        counts = [number_counts[number] for number in sorted_numbers]

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.bar(numbers, counts, color='skyblue')
        plt.xlabel('Numbers')
        plt.ylabel('Frequency')
        plt.title('Frequency Histogram')
        plt.savefig(output_file)  # Save the histogram as a PNG file
