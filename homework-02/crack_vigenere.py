#Текстовая ячейка <undefined>
# %% [markdown]
# Взлом шифра Виженера

#Кодовая ячейка <undefined>
# %% [code]
# -*- coding: utf-8 -*-
import string
from collections import Counter

# ============================================================
#  АНАЛИЗ ЧАСТОТНОСТИ АНГЛИЙСКОГО ЯЗЫКА
# ============================================================

LETTER_PROB = {
    'a': 8.167, 'b': 1.492, 'c': 2.782, 'd': 4.253, 'e': 12.702,
    'f': 2.228, 'g': 2.015, 'h': 6.094, 'i': 6.966, 'j': 0.153,
    'k': 0.772, 'l': 4.025, 'm': 2.406, 'n': 6.749, 'o': 7.507,
    'p': 1.929, 'q': 0.095, 'r': 5.987, 's': 6.327, 't': 9.056,
    'u': 2.758, 'v': 0.978, 'w': 2.360, 'x': 0.150, 'y': 1.974,
    'z': 0.074
}

BIGRAM_STATS = {
    'th': 3.56, 'he': 3.07, 'in': 2.43, 'er': 2.05, 'an': 1.99,
    're': 1.85, 'on': 1.76, 'en': 1.75, 'at': 1.49, 'es': 1.45,
    'ed': 1.45, 'it': 1.43, 'ou': 1.40, 'ha': 1.30, 'to': 1.28,
    'or': 1.28, 'is': 1.28, 'hi': 1.27, 'ng': 1.20, 'ar': 1.13,
    'te': 1.11, 'ti': 1.11, 'as': 1.10, 'nd': 1.10, 'of': 1.08,
    'st': 1.05, 'nt': 1.04, 'le': 1.01, 'io': 1.00, 've': 0.99,
    'co': 0.96, 'me': 0.96, 'de': 0.95, 'ro': 0.93, 'li': 0.92,
    'ri': 0.92, 'al': 0.89, 'se': 0.89, 'si': 0.88, 'om': 0.88,
    'ra': 0.88, 'ic': 0.85, 'ne': 0.85, 'la': 0.83, 'il': 0.82,
    'no': 0.82, 'ns': 0.82, 'be': 0.81, 'wi': 0.80, 'di': 0.79,
}


# ============================================================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def load_data(filename):
    with open(filename, 'r', encoding='cp1252') as f:
        return f.read()


def save_result(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(data)


def extract_letters_only(text):
    return ''.join(ch for ch in text if ch.isalpha())


def shift_cypher(text, shift):
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord('a') if ch.islower() else ord('A')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return ''.join(result)


# ============================================================
#  ОСНОВНЫЕ КРИПТОАНАЛИТИЧЕСКИЕ ФУНКЦИИ
# ============================================================

def calc_ioc(segment):
    letters = extract_letters_only(segment).lower()
    n = len(letters)
    if n < 2:
        return 0.0
    
    freq = Counter(letters)
    return sum(cnt * (cnt - 1) for cnt in freq.values()) / (n * (n - 1))


def extract_columns(data, step, offset):
    chars = []
    for i in range(offset, len(data), step):
        if data[i].isalpha():
            chars.append(data[i])
    return ''.join(chars)


def chi2_test(segment):
    letters = extract_letters_only(segment).lower()
    n = len(letters)
    if n == 0:
        return float('inf')
    
    freq = Counter(letters)
    chi2 = 0.0
    for ch in string.ascii_lowercase:
        observed = freq.get(ch, 0)
        expected = n * LETTER_PROB[ch] / 100
        if expected > 0:
            chi2 += (observed - expected) ** 2 / expected
    return chi2


def bigram_quality(text):
    clean = extract_letters_only(text).lower()
    if len(clean) < 2:
        return 0.0
    
    bigrams = [clean[i:i+2] for i in range(len(clean) - 1)]
    score = sum(BIGRAM_STATS.get(bg, 0.0) for bg in bigrams)
    return score / len(bigrams)


def guess_key_length(encrypted, max_len=30):
    results = []
    for length in range(1, max_len + 1):
        cols = [extract_columns(encrypted, length, i) for i in range(length)]
        avg_ioc = sum(calc_ioc(col) for col in cols) / length
        results.append((length, avg_ioc))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return [length for length, _ in results[:30]]


def recover_key(encrypted, key_len):
    key = ''
    for pos in range(key_len):
        column = extract_columns(encrypted, key_len, pos)
        candidates = []
        for letter in string.ascii_lowercase:
            decrypted = shift_cypher(column, -string.ascii_lowercase.index(letter))
            score = chi2_test(decrypted)
            candidates.append((score, letter))
        candidates.sort(key=lambda x: x[0])
        key += candidates[0][1]
    return key


def vigenere_decode(cipher, keyword):
    result = []
    key_len = len(keyword)
    key_pos = 0
    
    for ch in cipher:
        if ch.isalpha():
            shift = string.ascii_lowercase.index(keyword[key_pos % key_len])
            result.append(shift_cypher(ch, -shift))
            key_pos += 1
        else:
            result.append(ch)
            key_pos += 1
    
    return ''.join(result)


def minimize_keyword(keyword):
    kw_len = len(keyword)
    for divisor in range(1, kw_len // 2 + 1):
        if kw_len % divisor == 0:
            pattern = keyword[:divisor]
            if pattern * (kw_len // divisor) == keyword:
                return pattern
    return keyword


# ============================================================
#  ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    print("=" * 60)
    print("КРИПТОАНАЛИЗ ШИФРА ВИЖЕНЕРА")
    print("=" * 60)
    
    encrypted = load_data("cipher.txt")
    print(f"[+] Загружен файл: {len(encrypted)} символов")
    
    print("\n[1] Определение возможных длин ключа...")
    candidates = guess_key_length(encrypted)
    print(f"    Наиболее вероятные длины: {candidates[:15]}")
    
    print("\n[2] Перебор кандидатов и оценка качества...")
    results = []
    seen = set()
    
    for length in candidates:
        key = recover_key(encrypted, length)
        plain = vigenere_decode(encrypted, key)
        score = bigram_quality(plain)
        
        minimal = minimize_keyword(key)
        if minimal not in seen:
            seen.add(minimal)
            results.append((score, minimal, plain))
            print(f"    длина={length:2d}  ключ='{minimal}'  оценка={score:.5f}")
    
    results.sort(key=lambda x: x[0], reverse=True)
    
    print("\n[3] Лучший результат:")
    best_score, best_key, best_text = results[0]
    print(f"    Ключ: {best_key}")
    print(f"    Оценка: {best_score:.5f}")
    print(f"\n    Первые 300 символов:")
    print(f"    {best_text[:300]}")
    
    save_result(best_text, "cracked.txt")
    print("\n[+] Результат сохранён в 'cracked.txt'")
    
    return best_key, best_text


if __name__ == "__main__":
    main()
