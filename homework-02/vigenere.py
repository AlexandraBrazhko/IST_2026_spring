import string
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
    lower = string.ascii_lowercase
    upper = string.ascii_uppercase
    n = len(keyword)
    cipher_text = list(plaintext)
    for i in range(n):
        if keyword[i] in lower:
            shift = lower.find(keyword[i])
        else:
            shift = upper.find(keyword[i])
        # Fix: cipher_text[i::n] — это список символов, нужно конвертировать в строку и обратно
        segment = "".join(cipher_text[i::n])
        encrypted_segment = list(caesar.encrypt_caesar(segment, shift))
        cipher_text[i::n] = encrypted_segment
    return "".join(cipher_text)


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
    lower = string.ascii_lowercase
    upper = string.ascii_uppercase
    n = len(keyword)
    plaintext = list(ciphertext)
    for i in range(n):
        if keyword[i] in lower:
            shift = lower.find(keyword[i])
        else:
            shift = upper.find(keyword[i])
        # Fix: аналогично encrypt — конвертируем сегмент в строку
        segment = "".join(plaintext[i::n])
        decrypted_segment = list(caesar.encrypt_caesar(segment, -shift))
        plaintext[i::n] = decrypted_segment
    return "".join(plaintext)