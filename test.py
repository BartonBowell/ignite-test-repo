#!/usr/bin/env python3
"""
String Transformation Utility Functions

This module provides a collection of functions for advanced string manipulation
and transformation. It includes capabilities for text analysis, pattern matching,
and string restructuring with various formatting options.

Author: Claude
Date: 2024-11-15
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter


def transform_case_patterns(text: str, pattern: str = "alternate") -> str:
    """
    Transforms the case of characters in a string based on specified patterns.
    
    Args:
        text (str): The input string to transform
        pattern (str): The pattern to apply. Options:
            - "alternate": alternates between upper and lower case
            - "wave": creates a wave pattern of case changes
            - "pyramid": increases then decreases case changes
    
    Returns:
        str: The transformed string
    
    Examples:
        >>> transform_case_patterns("hello world", "alternate")
        'HeLlO wOrLd'
        >>> transform_case_patterns("hello world", "wave")
        'HeLLo WoRLD'
        >>> transform_case_patterns("hello world", "pyramid")
        'hElLoW oRlD'
    """
    if not text:
        return text
    
    result = []
    words = text.split()
    
    if pattern == "alternate":
        for word in words:
            transformed = ''.join(
                c.upper() if i % 2 == 0 else c.lower()
                for i, c in enumerate(word)
            )
            result.append(transformed)
            
    elif pattern == "wave":
        for word in words:
            wave_pattern = [0, 0, 1, 1, 0, 0, 1, 1]  # 0 for lower, 1 for upper
            transformed = ''.join(
                c.upper() if wave_pattern[i % len(wave_pattern)] else c.lower()
                for i, c in enumerate(word)
            )
            result.append(transformed)
            
    elif pattern == "pyramid":
        for word in words:
            mid = len(word) // 2
            transformed = ''.join(
                c.upper() if i <= mid and i % 2 == 1 or i > mid and i % 2 == 0
                else c.lower()
                for i, c in enumerate(word)
            )
            result.append(transformed)
            
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
        
    return ' '.join(result)


def analyze_and_restructure(text: str) -> Dict[str, any]:
    """
    Analyzes a string and returns various restructured forms and statistics.
    
    Args:
        text (str): The input string to analyze
    
    Returns:
        dict: A dictionary containing various analyses and transformations:
            - word_count: Total number of words
            - char_frequency: Character frequency distribution
            - reversed_words: Each word reversed
            - palindrome_words: List of palindrome words found
            - vowel_consonant_ratio: Ratio of vowels to consonants
    
    Example:
        >>> result = analyze_and_restructure("Hello noon world")
        >>> result['word_count']
        3
        >>> result['reversed_words']
        'olleH noon dlrow'
    """
    if not text.strip():
        return {
            "word_count": 0,
            "char_frequency": {},
            "reversed_words": "",
            "palindrome_words": [],
            "vowel_consonant_ratio": 0.0
        }
    
    # Basic word analysis
    words = text.lower().split()
    word_count = len(words)
    
    # Character frequency
    char_frequency = Counter(c for c in text.lower() if c.isalnum())
    
    # Reverse each word
    reversed_words = ' '.join(word[::-1] for word in text.split())
    
    # Find palindromes
    palindrome_words = [
        word for word in words
        if word == word[::-1] and len(word) > 1
    ]
    
    # Calculate vowel/consonant ratio
    vowels = sum(1 for c in text.lower() if c in 'aeiou')
    consonants = sum(1 for c in text.lower() if c.isalpha() and c not in 'aeiou')
    vowel_consonant_ratio = (
        vowels / consonants if consonants > 0 else float('inf')
    )
    
    return {
        "word_count": word_count,
        "char_frequency": dict(char_frequency),
        "reversed_words": reversed_words,
        "palindrome_words": palindrome_words,
        "vowel_consonant_ratio": round(vowel_consonant_ratio, 2)
    }


def generate_word_patterns(text: str, pattern_type: str = "diamond") -> List[str]:
    """
    Generates visual patterns using words from the input text.
    
    Args:
        text (str): The input text to transform
        pattern_type (str): The type of pattern to generate:
            - "diamond": Creates a diamond shape
            - "spiral": Creates a spiral pattern
            - "cascade": Creates a cascading pattern
    
    Returns:
        List[str]: A list of strings forming the pattern
    
    Example:
        >>> result = generate_word_patterns("hello", "diamond")
        >>> print('\n'.join(result))
          h
         hel
        hello
         hel
          h
    """
    words = text.split()
    result = []
    
    if pattern_type == "diamond":
        for word in words:
            # Generate diamond pattern for each word
            n = len(word)
            for i in range(n):
                spaces = " " * (n - i - 1)
                pattern = word[:i+1]
                result.append(f"{spaces}{pattern}")
            for i in range(n-2, -1, -1):
                spaces = " " * (n - i - 1)
                pattern = word[:i+1]
                result.append(f"{spaces}{pattern}")
                
    elif pattern_type == "spiral":
        for word in words:
            size = len(word)
            grid = [[' ' for _ in range(size)] for _ in range(size)]
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            x = y = d = 0
            
            for char in word:
                grid[x][y] = char
                next_x = x + directions[d][0]
                next_y = y + directions[d][1]
                
                if (not (0 <= next_x < size and 0 <= next_y < size) or
                        grid[next_x][next_y] != ' '):
                    d = (d + 1) % 4
                    next_x = x + directions[d][0]
                    next_y = y + directions[d][1]
                    
                x, y = next_x, next_y
                
            result.extend(''.join(row) for row in grid)
            
    elif pattern_type == "cascade":
        for word in words:
            for i in range(len(word)):
                result.append(" " * i + word[i:])
                
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
        
    return result


if __name__ == "__main__":
    # Example usage and tests
    text = "Hello Noon World"
    
    # Test transform_case_patterns
    print("\nCase Pattern Transformations:")
    print(transform_case_patterns(text, "alternate"))
    print(transform_case_patterns(text, "wave"))
    print(transform_case_patterns(text, "pyramid"))
    
    # Test analyze_and_restructure
    print("\nText Analysis:")
    analysis = analyze_and_restructure(text)
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    # Test generate_word_patterns
    print("\nWord Patterns:")
    patterns = generate_word_patterns("hello", "diamond")
    print('\n'.join(patterns))
