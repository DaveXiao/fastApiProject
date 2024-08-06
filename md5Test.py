import hashlib
from collections import OrderedDict

def md5_32(in_str):
    """MD5 hash function (32-bit)
    Converts the input string to a 32-bit MD5 hash.

    Args:
        in_str (str): The string to be hashed.

    Returns:
        str: The 32-bit MD5 hash of the input string. Returns an empty string if an exception occurs.
    """
    try:
        # Initialize the MD5 hash object
        md5 = hashlib.md5()

        # Update the hash object with the string encoded in UTF-8
        md5.update(in_str.encode('utf-8'))

        # Return the hexadecimal representation of the hash
        return md5.hexdigest()

    except Exception as e:
        # Print the exception information and return an empty string
        print(e)
        return ""

def remove_repeated_chars(s):
    """Removes repeated characters from a string.

    Args:
        s (str): The input string.

    Returns:
        str: The string with removed repeated characters.
    """
    # Use OrderedDict to preserve order and remove duplicates
    return "".join(OrderedDict.fromkeys(s))

def convert(in_str, secret_key):
    """Encryption/decryption algorithm.

    Args:
        in_str (str): The string to be encrypted or decrypted.
        secret_key (str): The key used for encryption/decryption.

    Returns:
        str: The result after encryption/decryption.
    """
    a = list(in_str)
    s = list(remove_repeated_chars(secret_key))

    for i in range(len(s)):
        for j in range(len(a)):
            a[j] = chr(ord(a[j]) ^ ord(s[i % len(s)]))

    return "".join(a)

def encode(str_to_encode):
    """Encodes a string into hexadecimal digits, suitable for all characters (including Chinese).

    Args:
        str_to_encode (str): The string to be encoded.

    Returns:
        str: The encoded string.
    """
    try:
        # Encode the string to UTF-8, then convert each byte to two hexadecimal digits
        return ''.join(f'{byte:02X}' for byte in str_to_encode.encode('utf-8'))
    except UnicodeEncodeError as e:
        print(e)
        return ""

def decode(hex_str):
    """Decodes a string from hexadecimal digits, suitable for all characters (including Chinese).

    Args:
        hex_str (str): The hexadecimal string to be decoded.

    Returns:
        str: The decoded string.
    """
    try:
        # Convert each pair of hexadecimal digits to a byte, then decode to UTF-8
        return bytes.fromhex(hex_str).decode('utf-8')
    except ValueError as e:
        print(e)
        return ""

def encrypt(in_str, secret_key):
    """Encrypts a string.

    Args:
        in_str (str): The original string.
        secret_key (str): The key used for encryption.

    Returns:
        str: The encrypted string.
    """
    hex_str = convert(in_str, secret_key)
    return encode(hex_str)

def decrypt(in_str, secret_key):
    """Decrypts a string.

    Args:
        in_str (str): The original string.
        secret_key (str): The key used for decryption.

    Returns:
        str: The decrypted string.
    """
    hex_str = decode(in_str)
    return convert(hex_str, secret_key)

if __name__ == "__main__":
    s = "12bvdde`中户人民共和国，1234@￥#%&*（）-=|+_}{[]/.,;:,.>》》。，《dkfjaskfaskdjfkdasj";
    # s = "123456"
    md5_hash = md5_32(s)
    encrypted = encrypt(s, "f8ee541137a2aa381abaac17886653ba")
    decrypted = decrypt(encrypted, "f8ee541137a2aa381abaac17886653ba")

    print("原始:", s)
    print("MD5后:", md5_hash)
    print("MD5后的长度:", len(md5_hash))
    print("加密的:", encrypted)
    print("加密后的长度:", len(encrypted))
    print("解密的:", decrypted)
