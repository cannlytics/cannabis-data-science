
"""
Types of queries:

RETRIEVAL_QUERY	Specifies the given text is a query in a search/retrieval setting.
SEMANTIC_SIMILARITY	Specifies the given text will be used for Semantic Textual Similarity (STS).
CLASSIFICATION	Specifies that the embeddings will be used for classification.

"""

# TODO: Read in the license data.
licenses_dir = r'C:\Users\keega\Documents\cannlytics\cannlytics\datasets\cannabis_licenses\data'
licenses_datafile = r"C:\Users\keega\Documents\cannlytics\cannlytics\datasets\cannabis_licenses\data\all\licenses-all-latest.csv"


# TODO: Try to find owners with similar names.


# TODO: Use text similarity (such as with gensim).


# TODO: Use OpenAI to group similar people.
prompt = """Given the list of businesses and business owners, can you 
group similarly named people together into organizations?

For example, given:

{"business": "Cannlytics", "owners": ["John Doe"]}
{"business": "Cannlytics, LLC", "owners": ["John L. Doe"]}
{"business": "Cannlytics, Inc.", "owners": ["John Doe"]}

Return:

{

    "organizations": [
        {
            "businesses": ["Cannlytics", "Cannlytics, LLC", "Cannlytics, Inc."],
            "owners": ["John Doe", "John L. Doe"]
        }
    ]

}

"""


# TODO: Find licenses owned by similar people.

