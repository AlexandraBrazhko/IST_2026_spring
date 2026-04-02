import caesar
def encrypt_vigenere(plaintext: str, keyword: str) -> str:
    """
    Encrypts plaintext using a Vigenere cipher.

    >>> encrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> encrypt_vigenere("python", "a")
    'python'
    >>> encrypt_vigenere("ATTACKATDAWN", "LEMON")
    'LXFOPVEFRNHR'
    """
    ciphertext = ""
    lower = string.ascii_lowercase
    upper = string.ascii_uppercase
    n = len(keyword)
    cipher_text = list(plaintext)
    for i in range(n):
      if keyword[i] in lower:
        shift = lower.find(keyword[i])
      else:
        shift = upper.find(keyword[i])
      cipher_text[i::n] = caesar.encrypt_caesar(plaintext[i::n], shift)
    cipher_text = "".join(cipher_text)
    return cipher_text


def decrypt_vigenere(ciphertext: str, keyword: str) -> str:
    """
    Decrypts a ciphertext using a Vigenere cipher.

    >>> decrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> decrypt_vigenere("python", "a")
    'python'
    >>> decrypt_vigenere("LXFOPVEFRNHR", "LEMON")
    'ATTACKATDAWN'
    """
    plaintext = ""
    lower = string.ascii_lowercase
    upper = string.ascii_uppercase
    n = len(keyword)
    plaintext = list(ciphertext)
    for i in range(n):
      if keyword[i] in lower:
        shift = lower.find(keyword[i])
      else:
        shift = upper.find(keyword[i])
      plaintext[i::n] = caesar.encrypt_caesar(ciphertext[i::n], -shift)
    plaintext = "".join(plaintext)
    return plaintext
