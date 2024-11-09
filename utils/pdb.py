import dataclasses
import pdb
@dataclasses.dataclass
class C:
    a: str       # 'a' has no default value
    b: int = 0   # assign a default value for 'b'
    def get_a(self):
        print(self.a)
pdb.set_trace()
c = C(a='10',b=1000)
c.get_a()