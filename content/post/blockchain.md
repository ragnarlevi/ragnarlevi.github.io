+++
title = "Proof-of-Work and Proof-of-Stake"
date = "2023-05-10"
author = "Ragnar Levi Gudmundarson"
tags = ["blockchain"]
+++


The purpose of this notebook is to look at proof of work blockchain and proof of stake blockchain in a very primitive manner.

We start with defining transactions and block objects for the PoW and PoS


```python
from dataclasses import dataclass
import json
from hashlib import sha256
from datetime import datetime
from dataclasses import asdict
import numpy as np
import time



@dataclass
class Transaction:
  sender: str
  receiver: str
  amount: float


@dataclass
class Pow_Block:
  version: int
  timestamp: float
  nonce: int
  difficulty: int
  previous_hash: str
  transactions: list[Transaction]



@dataclass
class Pos_Block:
  version: int
  timestamp: float
  validator: str
  previous_hash: str
  transactions: list[Transaction]
```

# Proof of Work

In proof of work, miners are competing in finding the so called nonce (number only used once) that satisfies a certain difficulty. If the nonce generates a number lower than the difficulty number, the nonce is valid and the miner who found it wins and can claim the rewards.

The nonce generates a hash that is converted to a number. Call that number $A$. The difficulty generates a number which we call $B$. If $A<B$, nonce is valid.



```python


HEX_BASE_TO_NUMBER = 16
DIFFICULTY = 1
#PREV_BLOCK_HASH = "68ffd13b24f9d73399a80aad9de06f676001fed56648314526cd23a4d18fef16"
TRANSACTIONS = (
    Transaction("sender1", "receiver1", 1),
    Transaction("sender2", "receiver2", 0.5),
    Transaction("sender3", "receiver3", 2.7),
)

def create_block_pow(nonce: int, difficulty: int, transactions: tuple, previous_hash) -> Pow_Block:
    cur_timestamp = datetime.now().timestamp()
    return Pow_Block(
        version=1,
        timestamp=cur_timestamp,
        nonce=nonce,
        previous_hash=previous_hash,
        difficulty=difficulty,
        transactions=transactions
    )

def encode_block_pow(block: Pow_Block) -> str:
    encoded_block = json.dumps(asdict(block)).encode()
    return sha256(sha256(encoded_block).digest()).hexdigest()

def calculate_hash_pow(nonce: int, transactions) -> str:
    block = encode_block_pow(nonce, DIFFICULTY, transactions)
    encoded_block_hash = encode_block_pow(block)
    return encoded_block_hash



```


```python



BYTE_IN_BITS = 256
HEX_BASE_TO_NUMBER = 16
SECONDS_TO_EXPIRE = 20


GENESIS_BLOCK = Pow_Block(
    version=1,
    timestamp=1231006505,
    difficulty=1,
    nonce=42,
    previous_hash="000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f",
    transactions=(Transaction("Satoshi Nakamoto", "Satoshi Nakamoto", 50),)
)


def calculate_difficulty_target(difficulty_bits: int) -> int:
    return 2 ** (BYTE_IN_BITS - difficulty_bits)


class Pow_Blockchain:
    VERSION = 1
    DIFFICULTY = 10
    MINUTES_TOLERANCE = 1

    def __init__(self):
        self.chain = [GENESIS_BLOCK]

    def get_last_block(self) -> Pow_Block:
        return self.chain[-1]
    
    def add_block(self, block: Pow_Block) -> bool:
        is_valid = self._validate_block(block)
        if is_valid:
            self.chain.append(block)

        return is_valid
    
    def get_difficulty(self) -> int:
        return self.DIFFICULTY


    def _validate_block(self, candidate: Pow_Block) -> bool:
        if candidate.version != self.VERSION:
            return False

        last_block = self.get_last_block()
        if candidate.previous_hash != encode_block_pow(last_block):
            return False
        
        if candidate.difficulty != self.DIFFICULTY:
            return False


        candidate_hash = encode_block_pow(candidate)
        candidate_decimal = int(candidate_hash, HEX_BASE_TO_NUMBER)

        target = calculate_difficulty_target(self.DIFFICULTY)
        is_block_valid = candidate_decimal < target

        return is_block_valid
    



def mine_proof_of_work(nonce: int, difficulty: int, prev_hash: str, transactions:tuple) -> tuple[bool, Pow_Block]:
    block = create_block_pow(nonce, difficulty, transactions, prev_hash)
    encoded_block = encode_block_pow(block)
    block_encoded_as_number = int(encoded_block, HEX_BASE_TO_NUMBER)
    decimal_target = calculate_difficulty_target(difficulty)

    if block_encoded_as_number < decimal_target:
        return True, block

    return False, block

```

Start blockchain


```python
import numpy as np
pow_blockchain = Pow_Blockchain()


nonce = 0
start_time = time.time()
found = False
prev_hash = encode_block_pow(pow_blockchain.get_last_block())

for i in range(10):
    nonce = 0
    start_time = time.time()
    found = False
    prev_hash = encode_block_pow(pow_blockchain.get_last_block())
    while not found:
        found, block = mine_proof_of_work(nonce, pow_blockchain.get_difficulty(), prev_hash, Transaction("Satoshi Nakamoto", "Satoshi Nakamoto", np.random.uniform()))

        if found:
            pow_blockchain.add_block(block)
        else:
            nonce += 1

        end_time = time.time()
        elapsed_time = end_time - start_time

        if elapsed_time > SECONDS_TO_EXPIRE:
            raise TimeoutError(
                f"Couldn't find a block within the given timeframe"
            )


```


```python
for block in pow_blockchain.chain:
    print(block)
    print("\n")
```

    Pow_Block(version=1, timestamp=1231006505, nonce=42, difficulty=1, previous_hash='000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f', transactions=(Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=50),))
    
    
    Pow_Block(version=1, timestamp=1683676835.03425, nonce=1271, difficulty=10, previous_hash='629982432748d68179f7f81e85b67925a59fb2b5082eeb5f6fc9257028204ea4', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=0.04643169129473279))
    
    
    Pow_Block(version=1, timestamp=1683676835.045253, nonce=533, difficulty=10, previous_hash='003e03f4189866a540edccaf5a8854a5f611a7707ff9c0df572c41f1e098fb03', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=0.23683450574403841))
    
    
    Pow_Block(version=1, timestamp=1683676835.091262, nonce=2219, difficulty=10, previous_hash='003abd0d23f25fc664171fbc7f6a25998ec28ee204b31cd45b74ff2c0ccf4309', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=0.30504848451670386))
    
    
    Pow_Block(version=1, timestamp=1683676835.103265, nonce=580, difficulty=10, previous_hash='00274768d89c7f4372eb15ff34a3c0c92da8f7acc98d37bce46c0dfbbcefad61', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=0.8891598058714782))
    
    
    Pow_Block(version=1, timestamp=1683676835.105266, nonce=96, difficulty=10, previous_hash='0033bfbdecf964db73b45fcdf19b0255143799583a4def154570cd1f859e8358', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=0.21893269713324048))
    
    
    Pow_Block(version=1, timestamp=1683676835.12527, nonce=993, difficulty=10, previous_hash='001436dd4a0710ee2dc4f443fe3ae851eaec7df44f77afdd4f86785fcfa197b4', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=0.13291033703083))
    
    
    Pow_Block(version=1, timestamp=1683676835.135273, nonce=486, difficulty=10, previous_hash='003037d579bb7aeef3befb4c281050ed0d2703d7a70b847b4f471e8417d59d59', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=0.138879958758775))
    
    
    Pow_Block(version=1, timestamp=1683676835.162279, nonce=1286, difficulty=10, previous_hash='000453475413a3c6c46c7bfa7c8a63fe5b068ed9d1b1e80a213d9dc166d220f8', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=0.6620923154562147))
    
    
    Pow_Block(version=1, timestamp=1683676835.189285, nonce=1339, difficulty=10, previous_hash='0020b8db96297591627d09101ab738fd167e3d98821c6200e60915366e8adc47', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=0.24158860154623452))
    
    
    Pow_Block(version=1, timestamp=1683676835.198287, nonce=447, difficulty=10, previous_hash='002b019a97422540e7523a898a40c451f7aa9acd1796fdcd0a5fc2a1e3d97d78', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=0.5398827320781564))
    
    
    

# Proof of Stake

In Proof of Stake, miners/validators are required to stake their tokens/balance in order to be chosen as the next block creator. Therefore, the miner that stakes the most amount of its currency has the highest chance of being chosen as the leader and creating the next block.

Compared to Proof of Work, where miners compete with each other in terms of computation power, here they compete in terms of currency.


```python
GENESIS_BLOCK_POS = Pos_Block(
    version=1,
    timestamp=1231006505,
    previous_hash="000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f",
    validator='Louie',
    transactions=Transaction('','',0)
)

def encode_block_pos(block):
    """
    calcuate block sha256 hash value
    """
    record  = "".join([
        str(block.version),
        str(block.timestamp),
        block.validator,
        block.previous_hash
    ])
    return sha256(record.encode()).hexdigest()



class Pos_Blockchain:
    VERSION = 1

    def __init__(self):
        self.chain = [GENESIS_BLOCK_POS]

    def get_last_block(self) -> Pos_Block:
        return self.chain[-1]
    
    def add_block(self, block: Pos_Block) -> bool:
        is_valid = self._validate_block(block)
        if is_valid:
            self.chain.append(block)

        return is_valid
    

    def get_difficulty(self) -> int:
        return self.DIFFICULTY


    def _validate_block(self, candidate: Pos_Block) -> bool:
        if candidate.version != self.VERSION:
            return False

        last_block = self.get_last_block()
        if candidate.previous_hash != encode_block_pos(last_block):
            return False
        



        return True








def create_block_pos(transactions: tuple, validator: str, previous_hash) -> Pow_Block:
        cur_timestamp = datetime.now().timestamp()
        return Pos_Block(
            version=1,
            timestamp=cur_timestamp,
            validator=validator,
            previous_hash=previous_hash,
            transactions=transactions
        )

def lottery_proof_of_stake(validators, block, prev_hash, transactions):
    participants = [user['id'] for user in validators]
    participants_amt = np.array([user['stake'] for user in validators])
    total_amt = np.sum(participants_amt)
    winner = np.random.choice(a = participants, p = participants_amt/total_amt)
    block = create_block_pos(transactions, winner, prev_hash)
    return block

    
    




```


```python
validators = [{'id':'Huey', 'stake':33}, {'id':'Dewey', 'stake':10}, {'id':'Louie', 'stake':99}]


pos_blockchain = Pos_Blockchain()


start_time = time.time()
found = False
prev_hash = encode_block_pos(pos_blockchain.get_last_block())

for i in range(10):
    prev_hash = encode_block_pos(pos_blockchain.get_last_block()) 
    block = lottery_proof_of_stake(validators, block, prev_hash, Transaction("Satoshi Nakamoto", "Satoshi Nakamoto", 100*np.random.uniform()))
    pos_blockchain.add_block(block)


for block in pos_blockchain.chain:
    print(block)
    print("\n")


```

    Pos_Block(version=1, timestamp=1231006505, validator='Louie', previous_hash='000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f', transactions=Transaction(sender='', receiver='', amount=0))
    
    
    Pos_Block(version=1, timestamp=1683676866.845412, validator='Dewey', previous_hash='df4413f8c82d31a43997633f241f3969788d7870410653a04abfa246335dd429', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=74.1256835500038))
    
    
    Pos_Block(version=1, timestamp=1683676866.845412, validator='Louie', previous_hash='b94d10d3a11af4e0c95b62f46883935fdccd761924ef849c99805f45ff99fb0f', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=24.151181171183932))
    
    
    Pos_Block(version=1, timestamp=1683676866.845412, validator='Louie', previous_hash='c4886eca584e50da1dd120fe8d242b1d06eb295361974d00565fa175e87674c4', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=79.4071220119231))
    
    
    Pos_Block(version=1, timestamp=1683676866.845412, validator='Louie', previous_hash='427e88cae6b25a7e648efced676cea59cbbf1f6141fee8cc17884b2dcc3905f3', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=35.83240492783906))
    
    
    Pos_Block(version=1, timestamp=1683676866.845412, validator='Louie', previous_hash='175e1d1f82a454f73cab7175bd9ccd1f69bab8a31e1bcb0b5437768a15e74441', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=64.68286159916468))
    
    
    Pos_Block(version=1, timestamp=1683676866.845412, validator='Huey', previous_hash='3ea4c3fbc5ccd3e7a513717705874f06d3db4d70ca33938cb546239d55dcfdad', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=66.83634658085363))
    
    
    Pos_Block(version=1, timestamp=1683676866.845412, validator='Louie', previous_hash='78bf9ae2cc5b67a8f4f773820fed55bf6c6036c7f3687d60f9f14930e6588d89', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=66.79188204734481))
    
    
    Pos_Block(version=1, timestamp=1683676866.845412, validator='Louie', previous_hash='2202892e13eb974736cfbf9e77138f7142996ab0ebc3b572571e7596eeef6070', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=77.54172125698501))
    
    
    Pos_Block(version=1, timestamp=1683676866.845412, validator='Louie', previous_hash='ea9e218d1e7dbdca541796500a13b80e2e039f73a04f17080a16683b74fc2e04', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=15.281043236113367))
    
    
    Pos_Block(version=1, timestamp=1683676866.845412, validator='Huey', previous_hash='3ba3b8089dc65d0b1336aa60e7e854d39d035150dc728f735737642e0b4b9bce', transactions=Transaction(sender='Satoshi Nakamoto', receiver='Satoshi Nakamoto', amount=15.538494296269745))
    
    
    
