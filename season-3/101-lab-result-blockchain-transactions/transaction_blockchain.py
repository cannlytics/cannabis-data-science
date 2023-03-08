"""
Lab Result Transactions Blockchain | Cannabis Traceability System
Copyright (c) 2023 Cannlytics
Copyright (c) 2020 Michael

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 2/22/2023
Updated: 2/22/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
License: <https://github.com/mchrupcala/blockchain-walkthrough/blob/master/LICENSE>
"""
# Standard imports:
import hashlib
import json
from time import time


class Blockchain(object):
    def __init__(self):
        # Blockchain data.
        self.chain = []

        # Pending transactions.
        self.pending_transactions = []

        # Genesis block.
        self.new_block(
            previous_hash='Cannlytics 22/Feb/2023 Cannabis traceability system.',
            proof=100,
        )

    def new_block(self, proof, previous_hash=None):
        """Create a new block listing key/value pairs of block information in a
        JSON object. Then reset the list of pending transactions and
        append the newest block to the chain.
        """
        # Create a new block.
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.pending_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }

        # Reset pending transactions.
        self.pending_transactions = []

        # Add the new block.
        self.chain.append(block)
        return block


    @property
    def last_block(self):
        """Search the blockchain for the most recent block."""
        return self.chain[-1]


    def new_transaction(self, sender, recipient, amount):
        """Add a transaction with data to the list of pending transactions.
        """
        # Create a transaction.
        transaction = {
            'sender': sender,
            'recipient': recipient,
            'amount': amount
        }

        # Add the transaction to pending transactions.
        self.pending_transactions.append(transaction)

        # Increment the block index.
        return self.last_block['index'] + 1


    def hash(self, block):
        """Hash one block."""
        # Turn block into a string.
        string_object = json.dumps(block, sort_keys=True)

        # Turn the block string into Unicode for hashing.
        block_string = string_object.encode()

        # Hash with SHA256 encryption.
        raw_hash = hashlib.sha256(block_string)

        # Translate the Unicode into a hexidecimal string.
        hex_hash = raw_hash.hexdigest()
        return hex_hash


# === Test ===
if __name__ == '__main__':

    # Initialize a blockchain.
    blockchain = Blockchain()

    # Read lab results data.
    import pandas as pd
    data = pd.read_excel('mcr-lab-results-2023-02-06T14-35-24.xlsx')

    # Sort by date.
    data.sort_values('date_tested', inplace=True)

    # Add transactions to the blockchain.
    for index, row in data[:5].iterrows():
        producer = row['producer']
        lab = row['lab']
        observation = row.to_json()
        transaction = blockchain.new_transaction(lab, producer, observation)

    # Create a new block.
    blockchain.new_block(12345)

    # Add additional transactions to the blockchain.
    for index, row in data[5:10].iterrows():
        producer = row['producer']
        lab = row['lab']
        observation = row.to_json()
        transaction = blockchain.new_transaction(lab, producer, observation)

    # Create a new block.
    blockchain.new_block(6789)

    # Rinse and repeat....!

print("Genesis block: ", blockchain.chain)
