

class Catalog:

    def __init__(self):

        self.boxsize = None
        self.scf = None
        self.h = None
        self.hr_dm = None
        self.box = None

        self.data = dict()

        return

    def __repr__(self):
        out = f'Box Number: {self.box}\n' \
              f'Keys: {[key for key in self.data.keys()]}\n' \
              f'Boxsize = {self.boxsize}\n' \
              f'h = {self.h}\n' \
              f'scf = {self.scf}'
        return out

    def __contains__(self, attr):
        return attr in self.data

    def __getitem__(self, attr):
        return self.data[attr]

    def __setitem__(self, key, value):
        self.data[key] = value
        return

