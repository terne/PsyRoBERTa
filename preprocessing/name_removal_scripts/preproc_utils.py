import re
#from src.models.da_ner import BertNer # model class for DaNLP NER model trained on DaNE


def list_split(a, n):
    # to split notes if they are too long for NER
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def remap_nametokens(string_el):
    if any([x in string_el for x in ["First", "Middle", "Last"]]):
        # Replacing all prior [Firstname], [Middlename] and [Lastname] tokens with a new [Name] token
        string_el = re.sub(r'\[(First|Last|Middle)Name[0-9]*\]',"[Name]", string_el)
    return string_el


def regex_emails_numbers_etc(string_el):
    new_string = string_el

    # email adresses
    email_regex1 = r'\S*@\S*\s?'
    email_regex2 = r'\S*@\S*'

    # phone numbers - now redundant 
    #number_regex1 = r'\d\d\d\d\d\d\d\d'
    #number_regex2 = r'\d\d \d\d \d\d \d\d'

    # urls
    url_regex1 = r'http\S+'
    url_regex2 = r'www.\S+'

    #new_string = re.sub(number_regex1,"[8digitnumber]", new_string)
    #new_string = re.sub(number_regex2,"[8digitnumber]", new_string)
    new_string = re.sub(url_regex1, "[Url]", new_string)
    new_string = re.sub(url_regex2, "[Url]", new_string)
    new_string = re.sub(email_regex1, "[Email]", new_string)
    new_string = re.sub(email_regex2, "[Email]", new_string)
    return new_string


def recursive_lastnamesub(x, removed_lastnames, start_count=0):
    # Getting the (last) names that follow a name token
    lastnames = re.findall('(?:(?<=\[Name\])|(?<=\[NER_PER\]))\s([A-ZÆØÅ][A-ZÆØÅa-zæøå\-]+)', x)
    # add the lastnames to the lastname list, except "D-" and "Vej" that are frequently, and mistakenly, captured
    removed_lastnames.extend([i for i in lastnames if i not in ["D-", "Vej"]])
    #looping and updating string, substituting lastnames with tokens
    for ln in set(lastnames):
        if ln in x:
            x = re.sub(r'(?:(?<=\s)|(?<=^))%s(?=$|[^0-9a-zæøå])'%(ln), "[Name]", x)
    # Check if there are more last names
    lastnames = re.findall('(?:(?<=\[Name\])|(?<=\[NER_PER\]))\s([A-ZÆØÅ][A-ZÆØÅa-zæøå\-]+)', x)
    #if not, return the string, else run this function again, but a maximum of 10 times, which will be the maximum amount of "lastnames" accepted (might be high and could be reduced)
    if len(lastnames) == 0 or start_count>10:
        return x, removed_lastnames
    else:
        start_count+=1
        return recursive_lastnamesub(x, removed_lastnames,start_count=start_count)


def preprocess_wo_NER(string_el, names_list_external):
    
    """ Using first name lists and regex only """
    
    # first using regular expressions to remove clearly defined patterns, phone numbers, urls and email adresses:
    new_string = regex_emails_numbers_etc(string_el)
    
    # replace all prior name tokens with a new token
    new_string = remap_nametokens(new_string)
    
    removed_firstnames = []
    
    for i in names_list_external:
        if i in new_string:
            removed_firstnames.extend(re.findall(r'(?:(?<=\s)|(?<=^)|(?<=\\))%s(?=$|[^0-9a-zA-ZæøåÆØÅ-])'%(i), new_string))
            # Replacing the ith name from the namelist with a [name] token, 
            # only if it starts a string or is after a space
            # and only if is is followed by end of string, punctuation or space (i.e. not directly followed by other characters)
            # This should make sure that we do not remove parts of words.
            new_string = re.sub(r'(?:(?<=\s)|(?<=^)|(?<=\\))%s(?=$|[^0-9a-zA-ZæøåÆØÅ-])'%(i), "[Name]", new_string)
            
    # now check if followed by more names        
    new_string,removed_lastnames = recursive_lastnamesub(new_string,[])
    
    return new_string, removed_firstnames, removed_lastnames


def lastnames_second_search(string_el, lastnames):
    x = string_el
    for ln in set(lastnames):
        if ln in x:
            # capture patterns "X. Lastname" and "Lastname, [name]", where [name] is as previously subbed firstname
            # and Lastname is a name previously identified as a likely lastname
            x = re.sub(r'[A-ZÆØÅ]. %s'%(ln), "[Name]", x)
            x = re.sub(r'%s,(?= \[Name\])'%(ln), "[Name]",x)
    return x