'''
Representing stocks and stock ownership
'''

class Stock:
    '''
    A single holding of a stock with symbol, shares, and price
    '''
    def __init__(self,name: str,shares: int,price: float):
        self.name = name
        self.shares = shares
        self.price = price

    def cost(self):
        return self.shares*self.price

    def sell(self,shares):
        self.shares = max(0, self.shares - shares)

    def __str__(self):
        out = f'Stock({self.name}, {self.shares}, {self.price})'
        return out

    def __repr__(self):
        out = f'Stock({repr(self.name)}, {self.shares}, {self.price})'
        return out
