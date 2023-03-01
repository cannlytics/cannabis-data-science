"""
Lab Result Transactions | Cannabis Traceability System
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 2/22/2023
Updated: 2/22/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

References:

    - Cannabis Tests
    URL: <https://huggingface.co/datasets/cannlytics/cannabis_tests>

    - MCR Labs Results 2023-02-06
    URL: <https://cannlytics.page.link/mcr-lab-results-2023-02-06>

Data Sources:

    - MCR Labs Test Results
    URL: <https://reports.mcrlabs.com>

    - PSI Labs Test Results
    URL: <https://results.psilabs.org/test-results/>

    - SC Labs Test Results
    <https://client.sclabs.com/>

    - Washington State Lab Test Results
    URL: <https://lcb.box.com/s/xseghpsq2t4i1musxj6mgd7b8rhxe7bm>

"""
# Internal imports:
import hashlib, secrets
import json

# External imports:
import ecdsa
from pydantic import BaseModel


class LabResult(BaseModel):
    """A data class representing a cannabis lab result."""
    sample_hash: str
    results_hash: str
    sample_id: str
    product_name: str
    producer: str
    product_type: str
    product_subtype: str
    date_tested: str
    analyses: str
    batch_number: str
    batch_size: str
    batch_units: str
    cannabinoids_method: str
    cannabinoids_status: str
    coa_algorithm: str
    coa_algorithm_entry_point: str
    coa_parsed_at: str
    coa_pdf: str
    coa_urls: str
    date_collected: str
    date_produced: str
    date_received: str
    date_retail: str
    delta_9_thc_per_unit: str
    distributor: str
    distributor_address: str
    distributor_city: str
    distributor_license_number: str
    distributor_state: str
    distributor_street: str
    distributor_zipcode: float
    foreign_matter_method: str
    foreign_matter_status: str
    heavy_metals_method: str
    heavy_metals_status: str
    images: str
    lab: str
    lab_address: str
    lab_city: str
    lab_county: str
    lab_email: str
    lab_id: str
    lab_image_url: str
    lab_latitude: float
    lab_license_number: str
    lab_longitude: float
    lab_phone: str
    lab_results_url: str
    lab_state: str
    lab_street: str
    lab_website: str
    lab_zipcode: int
    lims: str
    metrc_ids: str
    metrc_lab_id: str
    metrc_source_id: str
    microbes_method: str
    microbes_status: str
    moisture_content: str
    moisture_method: str
    mycotoxins_method: str
    mycotoxins_status: str
    notes: str
    pesticides_method: str
    pesticides_status: str
    producer_address: str
    producer_city: str
    producer_image_url: str
    producer_license_number: str
    producer_state: str
    producer_street: str
    producer_url: str
    producer_zipcode: float
    product_size: str
    public: float
    residual_solvents_method: str
    residual_solvents_status: str
    results: str
    sample_number: float
    sample_size: str
    sampling_method: str
    serving_size: str
    status: str
    sum_of_cannabinoids: float
    terpenes_method: str
    terpenes_status: str
    total_cannabinoids: float
    total_cbc: float
    total_cbd: float
    total_cbdv: float
    total_cbg: float
    total_terpenes: float
    total_terpenes_mg_g: float
    total_thc: float
    total_thcv: float
    url: str
    water_activity_method: str


# Print the schema.
schema = json.loads(LabResult.schema_json())
print(json.dumps(schema, indent=2))


#------------------------------------------------------------------------------
# Add lab result data to the blockchain.
# References:
# - https://github.com/nakov/Practical-Cryptography-for-Developers-Book/blob/master/digital-signatures/ecdsa-sign-verify-examples.md
# - https://github.com/richardkiss/pycoin
#------------------------------------------------------------------------------

# FIXME: Update `pycoin` example or find a better example.
# - 
# See: 
# See
# from pycoin.ecdsa.secp256k1 import secp256k1_generator, sign, verify
# from pycoin.ecdsa import generator_secp256k1, sign, verify
# from pycoin.serialize import b2h_rev, h2b, b2h
# from pycoin.tx.Tx import Tx, TxIn, TxOut
# from pycoin.tx.script import tools
# from pycoin.tx.pay_to import build_hash160_lookup
# from pycoin.encoding import EncodingError
# from pycoin.tx import Spendable


def sha3_256Hash(msg):
    """Hashing function."""
    hashBytes = hashlib.sha3_256(msg.encode('utf8')).digest()
    return int.from_bytes(hashBytes, byteorder='big')

# def signECDSAsecp256k1(msg, privKey):
#     "Signing function."
#     msgHash = sha3_256Hash(msg)
#     signature = sign(generator_secp256k1, privKey, msgHash)
#     return signature

# def verifyECDSAsecp256k1(msg, signature, pubKey):
#     """Signature verification"""
#     msgHash = sha3_256Hash(msg)
#     valid = verify(generator_secp256k1, pubKey, msgHash, signature)
#     return valid


# Generate a key pair for the sender and receiver
owner_1_private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1) 
owner_1_public_key = owner_1_private_key.get_verifying_key()
# sender_private_key = owner_1_private_key.ecdsa.Private_key( self.pubkey, secret )
# sender_public_key = ecdsa.ecdsa.Public_key( generator_secp256k1, generator_secp256k1 * secret )
# sender_private_key = secrets.randbelow(secp256k1_generator.order())
# sender_public_key = (secp256k1_generator * sender_private_key).pair()
# sender_private_key, sender_public_key = generate_secp256k1_keypair()

# TODO: Create owner 2 credentials.
# receiver_private_key, receiver_public_key = generate_secp256k1_keypair()

# Create a transaction.
transaction = b'{"coa_algorithm":"mcrlabs.py","coa_algorithm_entry_point":"parse_mcrlabs_coa","lims":"MCR Labs","url":"https:\\/\\/reports.mcrlabs.com","lab":"MCR Labs Massachusetts","lab_website":"https:\\/\\/mcrlabs.com","lab_license_number":"IL281277","lab_image_url":null,"lab_address":"85 Speen St. Lower Level, Framingham, MA 01701","lab_street":"85 Speen St. Lower Level","lab_city":"Framingham","lab_county":"Middlesex","lab_state":"MA","lab_zipcode":1701,"lab_latitude":42.31082,"lab_longitude":-71.38692,"lab_phone":"508-872-6666","lab_email":"hello@mcrlabs.com","product_name":"SFV OG x Chemdog D","producer":"Tron Extracts","product_type":"extract","total_cannabinoids":82.1,"total_terpenes":null,"date_tested":"2014-09-16T00:00:00","producer_url":null,"lab_results_url":"https:\\/\\/reports.mcrlabs.com\\/reports\\/52259","images":"[{\'url\': \'https:\\/\\/mcrlabs.com\\/wp-content\\/uploads\\/2014\\/09\\/S14-0568.png\', \'filename\': \'S14-0568.png\'}]","serving_size":null,"analyses":"[\'cannabinoids\']","results":"[{\'analysis\': \'cannabinoids\', \'key\': \'thca\', \'name\': \'THCa\', \'value\': 81.0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'cbd\', \'name\': \'CBD\', \'value\': 0.6, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'thc\', \'name\': \'THC\', \'value\': 0.3, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'cbn\', \'name\': \'CBN\', \'value\': 0.2, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'cbna\', \'name\': \'CBNa\', \'value\': 0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'cbdva\', \'name\': \'CBDva\', \'value\': 0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'cbg\', \'name\': \'CBG\', \'value\': 0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'cbga\', \'name\': \'CBGa\', \'value\': 0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'thcv\', \'name\': \'THCv\', \'value\': 0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'thcva\', \'name\': \'THCva\', \'value\': 0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'cbdv\', \'name\': \'CBDv\', \'value\': 0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'cbc\', \'name\': \'CBC\', \'value\': 0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'cbca\', \'name\': \'CBCa\', \'value\': 0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'cbl\', \'name\': \'CBL\', \'value\': 0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'cbla\', \'name\': \'CBLa\', \'value\': 0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'cbda\', \'name\': \'CBDa\', \'value\': 0, \'units\': \'percent\'}, {\'analysis\': \'cannabinoids\', \'key\': \'delta_8_thc\', \'name\': \'\\u03948-THC\', \'value\': 0, \'units\': \'percent\'}]","sample_id":"233de9e600ab59ebd853ea3b3f3b62a0151d2586fe51ce3d38fbe65829a1fb3f","coa_parsed_at":"2023-02-06T14:21:49.133034","cannabinoids_method":"High-Performance Liquid Chromatography","terpenes_method":null}'

# # Create a transaction input (spendable output from a previous transaction)
# tx_in = TxIn(b'\x00'*32, 0, b'', 0xffffffff)

# # Create a transaction output to the receiver
# receiver_address = receiver_public_key.address()
# tx_out = TxOut(100000, receiver_address.to_script())

# Add an OP_RETURN output with arbitrary data
# data = b'{"sample_hash": "test"}'
# op_return_script = tools.compile('OP_RETURN', data)
# tx_out2 = TxOut(0, op_return_script)
# tx = Tx(1, [tx_in], [tx_out, tx_out2], 0)

# Sign the transaction with the sender's private key.
owner_1_signature = owner_1_private_key.sign(transaction)
# tx.set_unspents([Spendable(100000, tx_in.script(), sender_private_key.public_copy())])
# lookup = build_hash160_lookup([sender_private_key, receiver_private_key])
# tx.sign(lookup)

# Print the raw transaction in hex format
owner_1_verification = owner_1_public_key.verify(
    owner_1_signature,
    transaction,
)
# print(b2h(tx.as_hex()))
