#!/usr/bin/env python3

class Param:

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

class Child(Param):

    def __repr__(self):
        print(self.a)
