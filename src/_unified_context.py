from .cache import Cache, TEXT_T, FLOAT_T
import pandas as pd
from .pagesystem import PageManager


class UnifiedContext(object):

    cache_schema = [("currentPage", TEXT_T)]

    def __init__(self, user: str) -> None:
        self.user = user
        # self.page_cache = page_cache
        self.cache = Cache.get_instance()
        self.user_data: pd.DataFrame = pd.DataFrame([])
        self.unified_context_id = "_id_UnifiedContextCoreCache"
        self.cache.InitCache(self.unified_context_id,
                             UnifiedContext.cache_schema)
        self.app_state: pd.DataFrame = self.cache.read_state_df(
            self.unified_context_id)
    # String type used to get forward usage of type names
    # https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class

    def SetCurrentPageManager(self, page_manager: "PageManager") -> None:
        # TODO Think about how to ensure this gets called before each render
        self.current_page_manager = page_manager

    def RestorePageState(self, page_manager_id: str) -> pd.DataFrame:
        print("Restore State")
        return {}

    def StorePageState(self, state: pd.DataFrame, page_manager_id: str) -> None:
        print("Store State")

    def CheckCacheExists(self, page_manager_id: str) -> bool:
        pass


UNIFIED_CONTEXT_CACHE_LOCATION = "__ucc/"


def LoadUnifiedContext() -> UnifiedContext:
    return UnifiedContext(user="Theko")
    # ucc = os.path.join(UNIFIED_CONTEXT_CACHE_LOCATION, "theko.json")
    # if not os.path.exists(UNIFIED_CONTEXT_CACHE_LOCATION):
    #     return UnifiedContext(user="theko")
    #     # os.mkdir(UNIFIED_CONTEXT_CACHE_LOCATION)
    # with open(ucc, 'r') as uccfile:
    #     uccjson = json.load(uccfile)
    #     return UnifiedContext(user="saved_user")


def StoreUnifiedContext(ctx: UnifiedContext) -> bool:
    pass
