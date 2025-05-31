import re
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import soundfile as sf

# Hebrew Package https://pypi.org/project/hebrew/ to install type:pip install hebrew
from hebrew import Hebrew
from hebrew.chars import HebrewChar, ALEPH
from hebrew import GematriaTypes
# Audio
from pydub import AudioSegment
from typing import List, Tuple, Dict,Union,Optional


# _translit_tables.py
signs_dict = {
    '%': 'אָחוּז',
    ',': 'פְּסִיק',
    '.': 'נְקֻדָּה'

}


num_dict_below_20={
    1: 'אֶחָד',
    2: 'שְׁנַיִם',
    3: 'שְׁלֹשָׁה',
    4: 'אַרְבָּעָה',
    5: 'חֲמִשָּׁה',
    6: 'שִׁשָּׁה',
    7: 'שִׁבְעָה',
    8: 'שְׁמוֹנָה',
    9: 'תִּשְׁעָה',
    10: 'עֶשֶׂר',
    11: 'אַחַד עָשָׂר',
    12: 'שְׁנֵים עָשָׂר',
    13: 'שְׁלֹשָׁה עָשָׂר',
    14: 'אַרְבָּעָה עָשָׂר',
    15: 'חֲמִשָּׁה עָשָׂר',
    16: 'שִׁשָּׁה עָשָׂר',
    17: "שִׁבְעָה עֶשְׂרֵה",
    18: "שְׁמוֹנָה עֶשְׂרֵה",
    19: "תִּשְׁעָה עֶשְׂרֵה"
}

num_dict_eq_above_20={
    20: "עֶשְׂרִים",
    30: "שְׁלשִׁים",
    40: "אַרְבָּעִים",
    50: "חֲמִשִּׁים",
    60: "שִׁשִּׁים",
    70: "שִׁבְעִים",
    80: "שְׁמוֹנִים",
    90: "תִּשְׁעִים",
    100: "מֵאָה",
    200: "מָאתַיִם",
    300: "שְׁלֹשׁ מֵאוֹת",
    400: "אַרְבָּעִ מֵאוֹת",
    500: "חֲמִשֶּׁ מֵאוֹת",
    600: "שֵׁשׁ מֵאוֹת",
    700: "שִׁבְעַ מֵאוֹת",
    800: "שְׁמוֹנֶ מֵאוֹת",
    900: "תִּשְׁעַ מֵאוֹת",
    1000: "אֶלֶף",
    2000: "אֲלַפַּיִם",
    3000: "שְׁלֹשֶׁת אֲלָפִים",
    4000: "אַרְבַּעַת אֲלָפִים",
    5000: "חֲמֵשׁ אֲלָפִים",
    6000: "שֵׁשׁ אֲלָפִים",
    7000: "שִׁבְעָה אֲלָפִים",
    8000: "שְׁמוֹנָה אֲלָפִים",
    9000: "תִּשְׁעָה אֲלָפִים"
}


 ##deals with . and , in a normal string
def break_to_letter_and_rebuild(text: str) -> List[str]: #✅
    """
    Splits the input string into segments, separating out '.' and ',' as their own tokens.
    
    Args:
        text (str): The input string to split.
    
    Returns:
        List[str]: List of tokens where words and punctuation ('.', ',') are separated.
    """
    return [token for token in re.split(r'([.,])', text) if token]

#print(break_to_letter_and_rebuild("שלום,מה קורה.היום")) 
# ➜ ['שלום', ',', 'מה קורה', '.', 'היום']


def breakdown(number: int) -> List[int]: #✅
    """
    Breaks a number into its component decimal parts.
    
    Example:
        123456 -> [100000, 20000, 3000, 400, 50, 6]
    
    Args:
        number (int): The number to decompose.
        
    Returns:
        List[int]: List of decimal components.
    """
    digits = []
    for place in [100000000, 10000000, 1000000, 100000, 10000, 1000, 100, 10, 1]:
        digit = (number // place) * place
        digits.append(digit)
        number %= place
    return digits

#print(breakdown(123456))
# ➜ [0, 0, 0, 100000, 20000, 3000, 400, 50, 6]

##auxilary function for number_to_hebrew , helps break down arrays of 3's
def build_three_num_heb(list_num, num_dict_below_20, num_dict_eq_above_20, last): #✅
    list_heb = []
    is_all_zero = True  # ישתנה אם אחד המספרים לא אפס

    hundreds, tens, units = list_num

    # מאות
    if hundreds != 0:
        is_all_zero = False
        list_heb.append(num_dict_eq_above_20[hundreds])

    # עשרות ואחדות קטנות מ-20 (למשל 11, 17 וכו')
    two_digit_number = tens * 10 + units
    is_combined = two_digit_number < 20 and tens != 0

    if is_combined:
        is_all_zero = False
        word = num_dict_below_20[two_digit_number]
        if hundreds != 0:
            word = "וְ" + word
        list_heb.append(word)
        return list_heb, int(is_all_zero == False)

    # עשרות (20, 30, ...)
    if tens != 0:
        is_all_zero = False
        word = num_dict_eq_above_20[tens * 10]
        list_heb.append(word)

    # אחדות
    if units != 0:
        is_all_zero = False
        word = num_dict_below_20[units]
        if hundreds != 0 or tens != 0 or last:
            word = "וְ" + word
        list_heb.append(word)

    return list_heb, int(is_all_zero == False)


# list_num = [3, 4, 2]  # 300 + 40 + 2
# last = True
# print(build_three_num_heb(list_num = list_num, num_dict_below_20 = num_dict_below_20, num_dict_eq_above_20 = num_dict_eq_above_20, last = last))
# ➜ ['שְׁלֹשׁ מֵאוֹת', 'וְשְׁנֵים עָשָׂר']
def NumberToHebrew(value): #✅
    # תווים מיוחדים
    signs_dict = {
        '%': 'אָחוּז',
        ',': 'פְּסִיק',
        '.': 'נְקֻדָּה'
    }

    # מילונים (בהתאם לגרסה הקודמת שלך)
    num_dict_below_20 = {
        1: 'אֶחָד',
        2: 'שְׁנַיִם',
        3: 'שְׁלֹשָׁה',
        4: 'אַרְבָּעָה',
        5: 'חֲמִשָּׁה',
        6: 'שִׁשָּׁה',
        7: 'שִׁבְעָה',
        8: 'שְׁמוֹנָה',
        9: 'תִּשְׁעָה',
        10: 'עֶשֶׂר',
        11: 'אַחַד עָשָׂר',
        12: 'שְׁנֵים עָשָׂר',
        13: 'שְׁלֹשָׁה עָשָׂר',
        14: 'אַרְבָּעָה עָשָׂר',
        15: 'חֲמִשָּׁה עָשָׂר',
        16: 'שִׁשָּׁה עָשָׂר',
        17: "שִׁבְעָה עֶשְׂרֵה",
        18: "שְׁמוֹנָה עֶשְׂרֵה",
        19: "תִּשְׁעָה עֶשְׂרֵה"
    }

    num_dict_eq_above_20 = {
        20: "עֶשְׂרִים",
        30: "שְׁלוֹשִׁים",
        40: "אַרְבָּעִים",
        50: "חֲמִשִּׁים",
        60: "שִׁשִּׁים",
        70: "שִׁבְעִים",
        80: "שְׁמוֹנִים",
        90: "תִּשְׁעִים",
        100: "מֵאָה",
        200: "מָאתַיִם",
        300: "שְׁלוֹשׁ מֵאוֹת",
        400: "אַרְבַּע מֵאוֹת",
        500: "חֲמֵשׁ מֵאוֹת",
        600: "שֵׁשׁ מֵאוֹת",
        700: "שִׁבְעַ מֵאוֹת",
        800: "שְׁמוֹנֶה מֵאוֹת",
        900: "תִּשְׁעַ מֵאוֹת"
    }

    # טיפול בקלט לא תקין
    if value is None or (isinstance(value, str) and not value.strip()):
        return []

    # טיפול בתווים מיוחדים (%, ., ,)
    if isinstance(value, str):
        value = value.strip()
        if value in signs_dict:
            return [signs_dict[value]]
        value = value.replace(',', '')
        try:
            value = float(value) if '.' in value else int(value)
        except:
            return ["שְׁגִיאָה"]

    # טיפול בעשרוני
    
    
    if (isinstance(value, float)):
        int_part = int(value)
        frac_part = str(value).split('.')[1]
        result = NumberToHebrew(int_part)
        result.append(signs_dict['.'])
        result.extend([NumberToHebrew(int(digit))[0] for digit in frac_part if digit != '0'])
        return result

    # אפס
    if value == 0:
        return ["אֶפֶס"]
    

    # מספרים קטנים (פחות מ־10,000) — ישירות
    if value < 10000:
        return build_small_number(value, num_dict_below_20, num_dict_eq_above_20)

    # מספרים גדולים יותר – פירוק שלישיות: מיליונים, אלפים, יחידות
    return build_large_number(value, num_dict_below_20, num_dict_eq_above_20)




# פונקציה לעיבוד מספרים קטנים
def build_small_number(number, dict_lt20, dict_ge20):
    result = []

    if number < 20:
        result.append(dict_lt20[number])
        return result

    if number < 100:
        tens = (number // 10) * 10
        ones = number % 10
        result.append(dict_ge20[tens])
        if ones:
            result.append("וְ" + dict_lt20[ones])
        return result

    if number < 1000:
        hundreds = (number // 100) * 100
        rest = number % 100
        result.append(dict_ge20[hundreds])
        if rest:
            rest_part = build_small_number(rest, dict_lt20, dict_ge20)
            if rest_part:
                rest_part[0] = "וְ" + rest_part[0]  # חיבור ו'
                result.extend(rest_part)
        return result

    # עבור מספרים עד 9999
    thousands = number // 1000
    rest = number % 1000
    if thousands == 1:
        result.append("אֶלֶף")
    else:
        result.append(dict_lt20.get(thousands, str(thousands)))
        result.append("אֲלָפִים")
    if rest:
        result.extend(build_small_number(rest, dict_lt20, dict_ge20))
    return result


# פונקציה לעיבוד מספרים גדולים (עם מיליונים ואלפים)
def build_large_number(number, dict_lt20, dict_ge20): #✅
    list_heb = []

    # פירוק לשלישיות
    str_num = str(number).zfill(9)
    s1 = [int(d) for d in str_num[0:3]]  # מיליונים
    s2 = [int(d) for d in str_num[3:6]]  # אלפים
    s3 = [int(d) for d in str_num[6:9]]  # יחידות

    for chunk, label, is_last in [(s1, "מִילְיוֹן", False), (s2, "אֶלֶף", False), (s3, "", True)]:
        group, has_value = build_three_num_heb(chunk, dict_lt20, dict_ge20, is_last)
        if has_value:
            list_heb.extend(group)
            if label:
                list_heb.append(label)

    return list_heb


# פונקציה לעיבוד שלישיית ספרות (היא זו שהייתה לך קודם)
def build_three_num_heb(digits, dict_lt20, dict_ge20, last): #✅
    hundreds, tens, ones = digits
    result = []

    if hundreds:
        result.append(dict_ge20.get(hundreds * 100, ""))

    two_digit = tens * 10 + ones
    if 0 < two_digit < 20:
        word = dict_lt20[two_digit]
        if hundreds:
            word = "וְ" + word
        result.append(word)
    else:
        if tens:
            result.append(dict_ge20.get(tens * 10, ""))
        if ones:
            word = dict_lt20[ones]
            if hundreds or tens or last:
                word = "וְ" + word
            result.append(word)

    return result, any([hundreds, tens, ones])

# print(" ".join(NumberToHebrew(5)))  
# print(" ".join(NumberToHebrew(100)))       # → מֵאָה
# print(" ".join(NumberToHebrew(12)))        # → שְׁנֵים עָשָׂר
# print(" ".join(NumberToHebrew("1,000")))   # → אֶלֶף
# print(" ".join(NumberToHebrew(123456789))) # → מֵאָה וְעֶשֶׂר מִילְיוֹן אַרְבָּעִים וְחֲמִשָּׁה אֶלֶף שֵׁשׁ מֵאוֹת שִׁבְעִים וְתִּשְׁעָה


##attempts to split string to number and string
def split_number_and_string(text): #✅
    """
    מפצלת מחרוזת לשלושה חלקים:
    לפני המספר הראשון, המספר עצמו, ואחרי המספר.
    אם לא נמצא מספר - מחזירה None.
    """
    match = re.search(r'\d+', text)
    if not match:
        return None

    number = match.group()
    start_index = match.start()
    end_index = match.end()

    before = text[:start_index]
    after = text[end_index:]

    return before, number, after

# tests = [
#     "item123value",
#     "abc456def789",
#     "onlytext",
#     "123start",
#     "end999",
#     "a1b2c3",
#     "before5after"
# ]

# for test in tests:
#     result = split_number_and_string(test)
#     print(f"Input: {test}")
#     print(f"Output: {result}")
#     print("---")
    
#######################################################################Auxilary functions

##check if string has number in it
def has_number(input_string): #✅
    # Use regular expression to search for any digits within the string
    return bool(re.search(r'\d', input_string))



##breaks text to list
def break_to_list(text): #✅
    """
    This function receives a string and returns a list of strings with each word from the input text.
    """
    lst = []
    for tav in text:
        lst.append(tav)
    return lst


########################################################################

def is_number_with_comma(string): #✅
    """
    בודקת האם מחרוזת היא מספר חוקי שכולל פסיק (כמו '1,000')
    או מספר שלם רגיל (כמו '1000').
    פסיקים בסוף או נקודות בסוף מוסרים לפני הבדיקה.
    """
    # הסרת פסיק או נקודה בסוף המחרוזת (למניעת שגיאה בהמרה)
    if string.endswith(',') or string.endswith('.'):
        string = string[:-1]

    # אם יש פסיק - בודקים שהוא בין שתי קבוצות של ספרות בלבד
    if ',' in string:
        parts = string.split(',')
        if len(parts) != 2:
            return False
        if not all(part.isdigit() for part in parts):
            return False
    # אם אין פסיק, נבדוק שהמחרוזת כולה מספר שלם
    elif not string.isdigit():
        return False

    return True

# print(is_number_with_comma("1,000"))     # True
# print(is_number_with_comma("1000"))      # True
# print(is_number_with_comma("1,00"))      # True (אם אתה רוצה לוודא שזה באמת פורמט אלפים – צריך שינוי)
# print(is_number_with_comma("1,000."))    # True
# print(is_number_with_comma("abc,123"))   # False
# print(is_number_with_comma("123."))      # True


def clean_number_with_comma(string): #✅
    """
    מנקה מחרוזת של מספרים:
    - מסירה פסיקים (למשל: '1,000' → '1000')
    - מסירה פסיק או נקודה מסוף המחרוזת
    - מחזירה את המספר כמספר שלם (int)
    """
    # הסרת פסיק או נקודה בסוף המחרוזת, אם יש
    if string.endswith(',') or string.endswith('.'):
        string = string[:-1]

    # הסרת כל הפסיקים מהמחרוזת
    string = string.replace(',', '')

    # המרה למספר שלם
    return int(string)

# print(clean_number_with_comma("1,000"))      # 1000
# print(clean_number_with_comma("10,000."))    # 10000
# print(clean_number_with_comma("500,"))       # 500


def is_number_with_decimal(string): #✅
    """
    בודקת אם המחרוזת מייצגת מספר עשרוני (float).
    מאפשרת פסיק בסוף (כמו לצורכי עיצוב), אך לא באמצע.
    """
    # אם יש פסיק בסוף – מסירים אותו
    if string.endswith(','):
        string = string[:-1]
    # אם יש פסיק במקום אחר – לא תקין
    elif ',' in string:
        return False

    # חובה לכלול נקודה עשרונית
    if '.' not in string:
        return False

    # בדיקה אם ניתן להמיר ל-float
    try:
        float(string)
        return True
    except ValueError:
        return False


# print(is_number_with_decimal("12.34"))     # True
# print(is_number_with_decimal("12.34,"))    # True
# print(is_number_with_decimal("12,34"))     # False
# print(is_number_with_decimal("1234"))      # False
# print(is_number_with_decimal("12.34.56"))  # False

def clean_decimal(string): #✅
    """
    מפצלת מחרוזת של מספר עשרוני לשני חלקים:
    - החלק השלם כ-int
    - החלק אחרי הנקודה כ-int (אם קיים), אחרת None

    תומכת בפסיק בסוף, אך לא באמצע.
    """
    # אם יש פסיק בסוף – הסר אותו
    if string.endswith(','):
        string = string[:-1]
    # אם יש פסיק במקום אחר – לא תקין
    elif ',' in string:
        return None

    # פיצול לפי נקודה עשרונית
    parts = string.split('.')

    try:
        whole = int(parts[0])
        decimal = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        return whole, decimal
    except (ValueError, IndexError):
        return None
    
    
# print(clean_decimal("123.45"))     # (123, 45)
# print(clean_decimal("123."))       # (123, None)
# print(clean_decimal("123"))       # (123, None)
# print(clean_decimal("123.45,"))    # (123, 45)
# print(clean_decimal("12,34"))      # None

def is_percentage(string): #✅
    """
    בודקת אם מחרוזת מייצגת ערך אחוז חוקי.
    תומכת במספרים שלמים או עשרוניים, עם או בלי פסיק/נקודה בסוף.
    לדוגמה: '50%', '12.5%', '100%,' ייחשבו תקינים.
    """
    # הסרת פסיק או נקודה מסוף המחרוזת, אם יש
    if string.endswith(',') or string.endswith('.'):
        string = string[:-1]

    # בדיקה אם המחרוזת מסתיימת באחוז
    if not string.endswith('%'):
        return False

    # הסרת סימן האחוז
    number_part = string[:-1].strip()

    # הסרת פסיק נוסף בסוף (אם הופיע לפני האחוז)
    if number_part.endswith(','):
        number_part = number_part[:-1]

    # בדיקה אם מה שנשאר הוא מספר חוקי
    try:
        float(number_part)
        return True
    except ValueError:
        return False


# print(is_percentage("50%"))       # True  
# print(is_percentage("12.5%"))     # True  
# print(is_percentage("100%,"))     # True  
# print(is_percentage("75.0%."))    # True  


def clean_percentage(string): #✅
    """
    מנקה מחרוזת שמייצגת אחוז, ומחזירה את המספר שבתוכה כמחרוזת:
    - מסירה תווים מיותרים (%, פסיק, נקודה בסוף)
    - מחזירה None אם הפורמט אינו תקין
    - דואגת להסיר אפסים מיותרים אחרי הנקודה
    """
    if not string:
        return None

    # הסרת פסיק או נקודה בסוף המחרוזת
    if string.endswith(',') or string.endswith('.'):
        string = string[:-1]

    # בדיקה וסילוק סימן האחוז
    if string.endswith('%'):
        string = string[:-1]
    else:
        return None

    # הסרת פסיק נוסף אם הופיע לפני סימן האחוז
    if string.endswith(','):
        string = string[:-1]

    # ניסיון להמיר למספר עשרוני
    try:
        number = float(string)
    except ValueError:
        return None

    # החזרת מחרוזת מספרית ללא אפסים מיותרים
    cleaned = str(number).rstrip('0').rstrip('.')
    return cleaned



# print(clean_percentage("50%"))       # '50'
# print(clean_percentage("12.0%"))     # '12'
# print(clean_percentage("12.50%"))    # '12.5'
# print(clean_percentage("12.0%,"))    # '12'
# print(clean_percentage("12.0%."))    # '12'
# print(clean_percentage("12%"))       # '12'
# print(clean_percentage("12%,"))      # '12'
# print(clean_percentage("abc%"))     # None


def is_number_range(string): #✅
    """
    בודקת אם מחרוזת מייצגת טווח מספרים תקני, למשל: '10-20'.
    תומכת בהסרת פסיק או נקודה מסוף המחרוזת.
    """
    if not string or '-' not in string:
        return False

    # הסרת פסיק או נקודה מהסוף, אם קיימים
    if string.endswith(',') or string.endswith('.'):
        string = string[:-1]

    # פיצול לפי מקף
    parts = string.split('-')

    # חייבים להיות בדיוק שני חלקים (מספרים לפני ואחרי המקף)
    if len(parts) != 2:
        return False

    # בדיקה שכל אחד מהחלקים הוא מספר שלם חיובי
    if not all(part.isdigit() for part in parts):
        return False

    return True

# print(is_number_range("10-20"))     # True  
# print(is_number_range("5-15,"))    # True  
# print(is_number_range("3-7."))    # True  
# print(is_number_range("10-"))       # False  
# print(is_number_range("-20"))      # False  
# print(is_number_range("10-abc"))    # False  


def clean_number_range(string): #✅
    """
    מנקה מחרוזת טווח מספרים ומחזירה זוג מספרים שלמים כ-tuple:
    לדוגמה: '10-20' → (10, 20)
    אם הפורמט אינו תקין – מחזירה None
    """
    if not string:
        return None

    # הסרת פסיק או נקודה מהסוף
    if string.endswith(',') or string.endswith('.'):
        string = string[:-1]

    # פיצול לפי מקף
    parts = string.split('-')
    if len(parts) != 2:
        return None

    try:
        start = int(parts[0])
        end = int(parts[1])
        return start, end
    except ValueError:
        return None


# print(clean_number_range("10-20"))     # (10, 20)
# print(clean_number_range("5-15,"))     # (5, 15)
# print(clean_number_range("8-12."))     # (8, 12)
# print(clean_number_range("10-"))       # None
# print(clean_number_range("a-b"))       # None

def is_pattern_number_with_heb(string): #✅
    """
    בודקת האם מחרוזת עומדת בתבנית:
    טקסט כלשהו בעברית (או תיאור) - ומספר תקני כלשהו.
    לדוגמה: 'סה"כ-100', 'עלות-1,000', 'ריבית-12.5%', 'טווח-10-20'
    """
    if not string or '-' not in string:
        return False

    # הסרת פסיק או נקודה מסוף המחרוזת
    if string.endswith(',') or string.endswith('.'):
        string = string[:-1]

    # פיצול לפי מקף ראשון בלבד
    parts = string.split('-', maxsplit=1)
    if len(parts) != 2:
        return False

    hebrew_part, numeric_part = parts

    # בדיקה אם החלק השני הוא אחד מהפורמטים המספריים התקינים
    if (
        numeric_part.isdigit()
        or is_number_range(numeric_part)
        or is_number_with_comma(numeric_part)
        or is_number_with_decimal(numeric_part)
        or is_percentage(numeric_part)
    ):
        return True

    return False


# print(is_pattern_number_with_heb("סה\"כ-100"))        # True  
# print(is_pattern_number_with_heb("סה\"כ-1,000"))      # True  
# print(is_pattern_number_with_heb("טווח-10-20"))       # True  
# print(is_pattern_number_with_heb("ריבית-12.5%"))      # True  


def clean_pattern_number_with_heb(string): #✅
    """
    מפצלת מחרוזת מהצורה <טקסט>-<מספר> או <טקסט>-<מספר-מספר>,
    ומחזירה tuple: (חלק הטקסט, ערך מנוקה, סוג הפורמט)

    תומך בפורמטים:
    - מספר עם פסיקים: '4,000'
    - מספר עשרוני: '2.9'
    - אחוזים: '12.5%'
    - טווח מספרים: '2000-2020'
    """
    if not string:
        return None

    # הסרת פסיק או נקודה מסוף המחרוזת
    if string.endswith(',') or string.endswith('.'):
        string = string[:-1]

    parts = string.split('-')

    # פורמט פשוט: טקסט-מספר
    if len(parts) == 2:
        label, number_part = parts[0], parts[1]

        if is_number_with_comma(number_part):
            return label, str(clean_number_with_comma(number_part)), "is_number_with_comma"

        elif is_number_with_decimal(number_part):
            whole, decimal = clean_decimal(number_part)
            return label, (whole, decimal), "is_number_with_decimal"

        elif is_percentage(number_part):
            return label, str(clean_percentage(number_part)), "is_percentage"

        elif number_part.isdigit():
            return label, int(number_part), "is_digit"

    # פורמט טווח: טקסט-מספר1-מספר2
    elif len(parts) == 3:
        label, num1, num2 = parts[0], parts[1], parts[2]
        range_string = f"{num1}-{num2}"

        if is_number_range(range_string):
            start, end = clean_number_range(range_string)
            return label, (start, end), "is_number_range"

    return None

# print(clean_pattern_number_with_heb("סה\"כ-4,000"))         # ('סה"כ', '4000', 'is_number_with_comma')
# print(clean_pattern_number_with_heb("ריבית-2.9"))           # ('ריבית', (2, 9), 'is_number_with_decimal')
# print(clean_pattern_number_with_heb("תשואה-4.5%"))          # ('תשואה', '4.5', 'is_percentage')
# print(clean_pattern_number_with_heb("שנים-1971-1972"))      # ('שנים', (1971, 1972), 'is_number_range')
# print(clean_pattern_number_with_heb("סה\"כ-1000"))          # ('סה"כ', 1000, 'is_digit')


def clean_number(word): #✅
    """
    מזהה פורמט מספרי מתוך מחרוזת ומחזיר ייצוג עברי (רשימת מילים).
    תומך: מספרים רגילים, עשרוניים, עם פסיקים, אחוזים, טווחים, ותבניות עם תחילית עברית.
    """
    # מספר עם פסיקים (אלפים וכו')
    if is_number_with_comma(word):
        return NumberToHebrew(int(clean_number_with_comma(word)))

    # מספר עשרוני
    elif is_number_with_decimal(word):
        part1, part2 = clean_decimal(word)
        return (
            NumberToHebrew(part1) +
            NumberToHebrew('.') +
            NumberToHebrew(part2)
        )

    # אחוזים
    elif is_percentage(word):
        part1, part2 = clean_decimal(clean_percentage(word))
        result = NumberToHebrew(part1)
        if part2 is not None:
            result += NumberToHebrew('.') + NumberToHebrew(part2)
        result += NumberToHebrew('%')
        return result

    # טווח מספרים (10-20)
    elif is_number_range(word):
        list_heb = []
        part1, part2 = clean_number_range(word)
        list_heb = NumberToHebrew(part1)
        list_heb.append("עַד")
        list_heb += NumberToHebrew(part2)
        return list_heb
       

    # תבנית בעברית עם מספר: לדוג' 'בְּ-100,000' או 'לְ-2.5%'
    elif is_pattern_number_with_heb(word):
        prefix, value, kind = (clean_pattern_number_with_heb(word))
        result = [prefix]

        if kind == "is_number_with_comma":
            result += NumberToHebrew(int(value))

        elif kind == "is_number_with_decimal":
            part1, part2 = value
            result += NumberToHebrew(part1) + NumberToHebrew('.') + NumberToHebrew(part2)

        elif kind == "is_percentage":
            part1, part2 = clean_decimal(value)
            result += NumberToHebrew(part1)
            if part2 is not None:
                result += NumberToHebrew('.') + NumberToHebrew(part2)
            result += NumberToHebrew('%')

        elif kind == "is_number_range":
            part1, part2 = value
            result += NumberToHebrew(int(part1)) + ["עַד"] + NumberToHebrew(int(part2))

        return result

    return None  # אם לא מזוהה כפלט תקני



# # רשימת בדיקות
# test_inputs = [
#     # "4,000",
#     # "2.9",
#     # "12.5%",
#     "100%,",
#     "3.0%.",
#     "10-20",
#     "1971-1972,",
#     "בְּ-100,000",
#     "מִ-0.7%",
#     "לְ-1.9",
#     "הַ-1,100",
#     "סה\"כ-4.5%",
#     "משהו-abc",     # לא תקין
#     "123",          # מספר רגיל - לא נתמך ישירות
# ]

# # הרצת הבדיקות
# for text in test_inputs:
#     result = clean_number(text)
#     print(f"input: {text}\noutput: {result}\n{'-'*40}")


##takes a letter in hebrew and returns the sound in english
def HebrewLetterToEnglishSound(obj,tzuptzik,last_letter=False): #✅
    obj = Hebrew(obj).string
    # map the nikud symbols to their corresponding phenoms
    nikud_map = {"ָ": "a", "ַ": "a", "ֶ": "e", "ֵ": "e", "ִ": "i", "ְ": "", "ֹ": "o", "ֻ": "oo", 'ּ': "", 'ֲ': 'a'}


    beged_kefet_shin_sin = {
        ############ B
        "בּ": "b",
        "בְּ": "b",
        "בִּ": "bi",
        "בֹּ": "bo",
        "בֵּ": "be",
        "בֶּ": "be",
        "בַּ": "ba",
        "בָּ": "ba",
        "בֻּ": "boo",
        ############ G
        "גּ": "g",
        "גְּ": "g",
        "גִּ": "gi",
        "גֹּ": "go",
        "גֵּ": "ge",
        "גֶּ": "ge",
        "גַּ": "ga",
        "גָּ": "ga",
        "גֻּ": "goo",
        ########### D
        "דּ": "d",
        "דְּ": "d",
        "דִּ": "di",
        "דֹּ": "do",
        "דֵּ": "de",
        "דֶּ": "de",
        "דַּ": "da",
        "דָּ": "da",
        "דֻּ": "doo",
        ########### K
        "כּ": "k",
        "כְּ": "k",
        "כִּ": "ki",
        "כֹּ": "ko",
        "כֵּ": "ke",
        "כֶּ": "ke",
        "כַּ": "ka",
        "כָּ": "ka",
        "כֻּ": "koo",
        ############ P
        "פּ": "p",
        "פְּ": "p",
        "פִּ": "pi",
        "פֹּ": "po",
        "פֵּ": "pe",
        "פֶּ": "pe",
        "פַּ": "pa",
        "פָּ": "pa",
        "פֻּ": "poo",
        ############ T
        "תּ": "t",
        "תְּ": "t",
        "תִּ": "ti",
        "תֹּ": "to",
        "תֵּ": "te",
        "תֶּ": "te",
        "תַּ": "ta",
        "תָּ": "ta",
        "תֻּ": "too",
        ############ S
        "שׂ": "s",
        "שְׂ": "s",
        "שִׂ": "si",
        "שֹׂ": "so",
        "שֵׂ": "se",
        "שֶׂ": "se",
        "שַׂ": "sa",
        "שָׂ": "sa",
        "שֻׂ": "soo",
        ########### SH
        "שׁ": "sh",
        "שְׁ": "sh",
        "שִׁ": "shi",
        "שֹׁ": "sho",
        "שֵׁ": "she",
        "שֶׁ": "she",
        "שַׁ": "sha",
        "שָׁ": "sha",
        "שֻׁ": "shoo",
    }

    vav = {
        "וֵּו": "ve",
        "וּ": "oo",
        "וּו": "oo",
        "וֹ": "o",
        "וֹו": "oo",
        "וְ": "ve",
        "וֱו": "ve",
        "וִ": "vi",
        "וִו": "vi",
        "וַ": "va",
        "וַו": "va",
        "וֶ": "ve",
        "וֶו": "ve",
        "וָ": "va",
        "וָו": "va",
        "וֻ": "oo",
        "וֻו": "oo"
    }


    letters_map = {
        "א": "",
        "ב": "v",
        "ג": "g",
        "ד": "d",
        "ה": "hh",
        "ו": "v",
        "ז": "z",
        "ח": "h",
        "ט": "t",
        "י": "y",
        "כ": "h",
        "ל": "l",
        "מ": "m",
        "נ": "n",
        "ס": "s",
        "ע": "",
        "פ": "f",
        "צ": "ts",
        "ק": "k",
        "ר": "r",
        "ש": "sh",
        "ת": "t",
        "ן": "n",
        "ם": "m",
        "ף": "f",
        "ץ": "ts",
        "ך": "h",
    }

    patah_ganav={
        "חַ": "ah",
        "חָ": "ah",
        "הַ": "hha",
        "הָ": "hha",
        "עַ": "a",
        "עָ": "a",

    }

    tzuptzik_letters={
        ##G
        "ג": "j",
        "גְ": "j",
        "גִ": "ji",
        "גֹ": "jo",
        "גֵ": "je",
        "גֶ": "je",
        "גַ": "ja",
        "גָ": "ja",
        "גֻ": "joo",
        "גּ": "j",
        "גְּ": "j",
        "גִּ": "ji",
        "גֹּ": "jo",
        "גֵּ": "je",
        "גֶּ": "je",
        "גַּ": "ja",
        "גָּ": "ja",
        "גֻּ": "joo",

        ##ch
        "צ": "ch",
        "צְ": "ch",
        "צִ": "chi",
        "צֹ": "cho",
        "צֵ": "che",
        "צֶ": "che",
        "צַ": "cha",
        "צָ": "cha",
        "צֻ": "choo",

        ##ch
        "ץ": "ch",
        "ץְ": "ch",
        "ץִ": "chi",
        "ץֹ": "cho",
        "ץֵ": "che",
        "ץֶ": "che",
        "ץַ": "cha",
        "ץָ": "cha",
        "ץֻ": "choo",

        ##Z
        "ז": "zh",
        "זְ": "zh",
        "זִ": "zhi",
        "זֹ": "zho",
        "זֵ": "zhe",
        "זֶ": "zhe",
        "זַ": "zha",
        "זָ": "zha",
        "זֻ": "zhoo",
    }

    if last_letter:
        if obj in patah_ganav:
            return patah_ganav[obj]

    if tzuptzik==True:
        if obj in tzuptzik_letters:
            return tzuptzik_letters[obj]

    if obj in beged_kefet_shin_sin:
        return beged_kefet_shin_sin[obj]
    elif obj in vav:
        return vav[obj]
    else:
        lst = break_to_list(obj)
        string = ""
        for item in lst:
            if item in letters_map:
                string += letters_map[item]
            if item in nikud_map:
                string += nikud_map[item]

        return string


def HebrewWordToEnglishSound(word, index): #✅
    """
    ממירה מילה בעברית לייצוג פונטי באנגלית.
    """
    hs = Hebrew(word)
    graphemes = list(hs.graphemes)
    letters = Hebrew(graphemes).string

    english_sound = []

    for i, letter in enumerate(letters):
        # בדיקת 'צופציק' (תו apostrophe אחרי האות הנוכחית)
        has_tzuptzik = (i < len(letters) - 1 and letters[i + 1] == "'")

        # האם זו האות האחרונה במילה
        is_last = (i == len(letters) - 1)

        # המרת האות לפונטיקה
        eng = HebrewLetterToEnglishSound(letter, has_tzuptzik, is_last)
        english_sound.append(eng)

    result = "".join(english_sound)

    # הסרת כפילות בסוף (yy → y), אם קיימת
    if result.endswith("yy"):
        result = result[:-1]

    return result


# print(HebrewWordToEnglishSound("שלום", 0))
# print(HebrewWordToEnglishSound("מים", 1))
# print(HebrewWordToEnglishSound("כלב", 2))
# print(HebrewWordToEnglishSound("תלמיד", 3))
# print(HebrewWordToEnglishSound("גיבור'", 4))
# print(HebrewWordToEnglishSound("ירושלים", 5))

def HebrewToEnglish(sentence, index=0): #✅
    """
    מקבלת משפט בעברית ומחזירה אותו כאותיות באנגלית לפי צלילים.
    """
    words = sentence.split()
    new_sentence = ""

    for word in words:
        # אם המילה לא כוללת מספר, מעבדים אותה רגיל
        if not has_number(word):
            # מפרקים את המילה לאותיות ולסימני פיסוק
            broken_word = break_to_letter_and_rebuild(word)

            for brk_word in broken_word:
                if brk_word in ['.', ',', ';']:
                    new_sentence += "q "  # מוסיפים "שקט" אחרי פיסוק
                else:
                    ret_sentence = HebrewWordToEnglishSound(brk_word, index)
                    new_sentence += ret_sentence + " "

        # אם המילה כוללת מספר
        else:
            try:
                before_num, num, after_num = split_number_and_string(word)

                # אם יש עוד מספרים לפני או אחרי – כנראה מחרוזת עם כמה מספרים, מפרקים אחד אחד
                if has_number(after_num) or has_number(before_num):
                    list_of_numbers = clean_number(word)
                    for number in list_of_numbers:
                        ret_sentence = HebrewWordToEnglishSound(number, index)
                        new_sentence += ret_sentence + " "

                else:
                    # קודם מתרגמים את החלק לפני המספר
                    if before_num:
                        ret_sentence = HebrewWordToEnglishSound(before_num, index)
                        new_sentence += ret_sentence + " "

                    # מתרגמים את המספר למילים בעברית ואז לפונמות באנגלית
                    num_digits = [s for s in word if s.isdigit()]
                    num_int = int("".join(num_digits))
                    list_of_numbers = NumberToHebrew(num_int)

                    for number in list_of_numbers:
                        ret_sentence = HebrewWordToEnglishSound(number, index)
                        new_sentence += ret_sentence + " "

                    # מתרגמים את החלק שאחרי המספר
                    if after_num:
                        ret_sentence = HebrewWordToEnglishSound(after_num, index)
                        new_sentence += ret_sentence + " "

            except Exception as e:
                print(f"שגיאה בפונקציה split_number_and_string בשורה {index} עם המילה: {word}")

    return new_sentence


if __name__ == "__main__":
    hebrew_sentence = "5, 4, 3, 2, 1"
    result = HebrewToEnglish(hebrew_sentence, index=1)
    print(result)