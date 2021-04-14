from typing import NewType

from .unified_context import UnifiedContext

# This is defined to avoid a cyclic dependancy between UnifiedContext and PageManager/Page
UnifiedContextType = NewType("UnifiedContextType", UnifiedContext)
