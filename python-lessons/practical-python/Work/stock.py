'''
Representing stocks and stock ownership
'''

class Stock:
    '''
    A single holding of a stock with symbol, shares, and price
    '''
    __slots__ = ('name','_shares','price')
    def __init__(self,name: str,shares: int,price: float):
        self.name = name
        self.shares = shares
        self.price = price

    @property
    def cost(self):
        return self.shares*self.price

    @property
    def shares(self):
        return self._shares

    @shares.setter
    def shares(self, value):
        if not isinstance(value, int):
            raise TypeError(f'Number of shares must be an int received a: {type(value)}')
        self._shares = value

    def sell(self,shares):
        self.shares = max(0, self.shares - shares)

    def __str__(self):
        out = f'Stock({self.name}, {self.shares}, {self.price})'
        return out

    def __repr__(self):
        out = f'Stock({repr(self.name)}, {self.shares}, {self.price})'
        return out
