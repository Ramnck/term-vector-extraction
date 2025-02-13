"""
This package contains modules for handling various types of documents and searching platforms APIs.
#### Documents:
- JSON 
- XML
- Blank (easy-modifiable template)
#### Searching Platforms APIs:
- Elasticsearch
- FIPS
- Filesystem (doesnt implement search yet)
"""

from .blank import *
from .es import *
from .fips import *
from .fs import *
from .xml import *
