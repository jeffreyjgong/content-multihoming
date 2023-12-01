class DID:
    def __init__(self, did, size, replicationRule):
        self.did = did
        self.size = size
        self.replicationRule = replicationRule

class RSE:
    def __init__(self, name, capacity, dids):
        self.name = name
        self.capacity = capacity
        self.dids = dids
        self.capacityLeft = capacity - sum([did.size for did in dids])
        self.capacityUsed = sum([did.size for did in dids])
        self.cloudCostModel = None

