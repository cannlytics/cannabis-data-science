"""
Cannabis Traceability System Blockchain
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 2/14/2023
Updated: 2/15/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Notes:

    A blockchain is a system where each transaction is timestamped,
    then hashed into an ongoing chain of hash-based proof-of-work.

    The proof-of-work system allows for a consensus on a single
    chronological history of the chain in the correct order in
    which the transactions were made.

Setup:

    Flask is used to create a simple API: pip install Flask

References:

    - How to Create a Simple Blockchain using Python
    https://www.section.io/engineering-education/how-to-create-a-blockchain-in-python/

    - Python Tutorial: Build A Blockchain In < 60 Lines of Code
    https://medium.com/coinmonks/python-tutorial-build-a-blockchain-713c706f6531

    - Create simple Blockchain using Python
    https://www.geeksforgeeks.org/create-simple-blockchain-using-python/

    - Understand bitcoin transaction json extracted from blockchain
    https://bitcoin.stackexchange.com/questions/105306/understand-bitcoin-transaction-json-extracted-from-blockchain

    - Bitcoin: A Peer-to-Peer Electronic Cash System
    https://bitcoin.org/bitcoin.pdf

    - Cannabis Regulatory Agency Summarily Suspends Licenses of Corunna Processor
    https://www.michigan.gov/lara/news-releases/2023/02/03/cannabis-regulatory-agency-summarily-suspends-licenses-of-corunna-processor

"""
# Standard imports:
import datetime
import hashlib
import json

# External imports: 
from flask import Flask, jsonify


class Blockchain:
 
    def __init__(self):
        """Initialize a Blockchain with a genesis block."""
        self.chain = []
        self.create_block(proof=1, previous_hash='0')
 
    def create_block(self, proof, previous_hash):
        """Add a block to the blockchain."""
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(datetime.datetime.now()),
            'proof': proof,
            'previous_hash': previous_hash,
        }
        self.chain.append(block)
        return block
 
    def print_previous_block(self):
        """Display the previous block."""
        return self.chain[-1]
 
    def proof_of_work(self, previous_proof):
        """Mine a new block upon successful proof-of-work."""
        new_proof = 1
        check_proof = False
 
        while check_proof is False:
            hash_operation = hashlib.sha256(
                str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:5] == '00000':
                check_proof = True
            else:
                new_proof += 1
 
        return new_proof
 
    def hash(self, block):
        """Create a hash of a block."""
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()
 
    def chain_valid(self, chain):
        """Determine if the chain is valid"""
        previous_block = chain[0]
        block_index = 1
 
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
 
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(
                str(proof**2 - previous_proof**2).encode()).hexdigest()
 
            if hash_operation[:5] != '00000':
                return False
            previous_block = block
            block_index += 1
 
        return True
 
 
# Create a simple API.
app = Flask(__name__)
 
# Create a Blockchain.
blockchain = Blockchain()
 
# API endpoint to mine a new block .
@app.route('/mine_block', methods=['GET'])
def mine_block():
    previous_block = blockchain.print_previous_block()
    previous_proof = previous_block['proof']
    proof = blockchain.proof_of_work(previous_proof)
    previous_hash = blockchain.hash(previous_block)
    block = blockchain.create_block(proof, previous_hash)
    response = {
        'message': 'A block is MINED',
        'index': block['index'],
        'timestamp': block['timestamp'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200
 

# API endpoint to display the blockchain in JSON format.
@app.route('/get_chain', methods=['GET'])
def display_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain)
    }
    return jsonify(response), 200
 

# API endpoint to check the validity of the blockchain.
@app.route('/valid', methods=['GET'])
def valid():
    valid = blockchain.chain_valid(blockchain.chain)
 
    if valid:
        response = {'message': 'The Blockchain is valid.'}
    else:
        response = {'message': 'The Blockchain is not valid.'}
    return jsonify(response), 200
 
 
# Run the server locally.
app.run(host='127.0.0.1', port=5000)
