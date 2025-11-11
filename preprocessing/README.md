
## Addtional de-identification of health professionals
Patients’ personal and sensitive information, including patients’ names, relatives’ names, addresses, social security numbers, phone numbers, and email addresses were first masked with exact matching (names and addresses on record) and then with pattern matching. Addresses were masked on exact matches of tokens (names and addresses split on whitespace), while social security numbers, phone numbers, emails and URLs were masked using pattern matching with Python regular expressions. We performed further masking of names of doctors and other health professionals (see ```/name_removal_scripts```); however, their names were unknown. Therefore, to mask the names of health professionals as well (avoid models’ relying on names as spurious features) we developed a method, similar to the approaches utilized in previous studies [1] that used lists of diverse names and Python regular expressions to identify them within the clinical notes. A list of first names was constructed from publicly available lists of both Danish and international first names. The Danish names consisted of names approved by the Danish Agency of Family Law (https://familieretshuset.dk/navne/navne/godkendte-fornavne, April 2023). International name lists were extracted from two independent GitHub repositories and consisted of names from a wide range of nationalities (https://github.com/tuqqu/gender-detector/blob/master/data/nam_dict.txt, https://github.com/pipinstallpip/gender/blob/master/data/sql/names.sql, April 2023). The extracted names were collected in one text file and reviewed for errors. Then, each first name on the list was replaced by the token ‘[Name]’ in string x (i.e. the clinical note) with a case-sensitive regular expression IF the name was at the beginning of the string OR after a whitespace OR after a backward slash (since it was common to sign with “\[Name]”), AND IF the name was followed by a whitespace, punctuation, or end of string: The extracted names were collected in one text file and reviewed for noticeable issues. Then, each first name on the list was replaced by the token ‘[Name]’ in string x (i.e. the clinical note) with a case-sensitive regular expression IF the name was at the beginning of the string OR after a whitespace OR after a backward slash (since it was common to sign with “\[Name]”), AND IF the name was followed by a whitespace, punctuation, or end of string: 

```python
re.sub(r'(?:(?<=\s)|(?<=^)|(?<=\\))%s(?=$|[^0-9a-zA-ZæøåÆØÅ-])'%(i), "[Name]", x)
```

A second regular expression was then used to recognize last names. Since we did not have lists of last names, these were instead identified purely by their pattern. A substring of string x was recognized as a last name IF it followed a ‘[Name]’ token and whitespace AND IF it started with an uppercase letter:

```python
re.findall('(?:(?<=\[Name\])\s([A-ZÆØÅ][A-ZÆØÅa-zæøå\-]+)', x)
```

A substring recognized as a last name, ln, was replaced by a ‘[Name]’ token IF the last name was at the beginning of the string OR after a whitespace AND IF it was followed by a whitespace, punctuation, OR end of string: 

```python
re.sub(r'(?:(?<=\s)|(?<=^))%s(?=$|[^0-9a-zæøå])'%(ln), "[Name]", x)
```

Last names were replaced recursively until there were no more patterns fitting the regular expression or the recursion had run more than 10 times. Replaced strings were saved in two text files, one for first names and one for last names. In the first few iterations, on a quarter of the data, these lists were reviewed for errors and the method and name lists were updated accordingly to reduce errors. Since last names often appeared without or before first names, in patterns such as “X. Lastname” and “Lastname, [Name]”, a final search was performed to replace the previously identified last names in these cases:

```python
re.sub(r'[A-ZÆØÅ]. %s'%(ln), "[Name]", x)

re.sub(r'%s,(?= \[Name\])'%(ln), "[Name]",x)
```

Strings used to mask PII, such as ‘[Name]’ and ‘[URL]’, were added as special tokens for the models’ respective tokenizers.

We had experimented with a publicly available Danish NER model[2] to compliment regular expression; however, the model did not transfer well to our clinical texts and was consequently unapplicable. In future directions, our de-identification may be improved by training PsyRoBERTa on our regular expression annotations, to teach it to recognize the remaining, similar patterns that escaped our rules.

### References
[1] Sundahl Laursen M, Skyttegaard Pedersen J, Søgaard Hansen R, Rajeeth Savarimuthu T, Just Vinholt P. Danish Clinical Named Entity Recognition and Relation Extraction. In: Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa). Association for Computational Linguistics; 2023:655-666. Accessed January 6, 2025. https://aclanthology.org/2023.nodalida-1.65/

[2] Hvingelby R, Pauli AB, Barrett M, Rosted C, Lidegaard LM, Søgaard A. DaNE: A Named Entity Resource for Danish. In: Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020). European Language Resources Association (ELRA); 2020:4597-4604. Accessed March 5, 2025. https://aclanthology.org/2020.lrec-1.565
