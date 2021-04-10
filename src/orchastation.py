from .cache import Cache, TEXT_T, FLOAT_T

import pandas as pd
from typing import Dict, List, NewType
from streamlit import error as st_error


class UnifiedContext(object):
    # Passes Cache into Page render calls
    cache_schema = [("currentPage", TEXT_T)]

    def __init__(self, user: str) -> None:
        self.user = user
        # self.page_cache = page_cache
        self.cache = Cache.get_instance(Cache.default_connection_url)
        self.user_data: pd.DataFrame = pd.DataFrame([])
        self.unified_context_id = "_id_UnifiedContextCoreCache"
        self.cache.InitCache(self.unified_context_id,
                             UnifiedContext.cache_schema)
        # self.app_state: pd.DataFrame = self.cache.read_state_df(
        #     self.unified_context_id)
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


class Page(object):

    def __init__(self, id: str, page_manager: "PageManager") -> None:
        self.id = id
        self.page_manager = page_manager
        # self.public_cache: CacheType = {}

    def RenderPage(self, ctx: UnifiedContext) -> None:
        raise NotImplementedError


class PageManager(object):
    __cache_schema = [("ticker", TEXT_T), ("allocation", FLOAT_T)]

    def __init__(self, page_manager_id: str, context: UnifiedContext):
        self.NO_PAGES = ""
        self.page_manager_id = page_manager_id
        self.cache_id = "_id_%s" % self.page_manager_id
        self.pages: Dict[str, Page] = {}
        self.current_page: str = self.NO_PAGES
        self.cache_df: pd.DataFrame = None
        # context.InitCache(self.cache_id, PageManager.cache_schema)
        self.context = context

    def RegisterPage(self, new_page_renderer: Page) -> None:
        assert(new_page_renderer.id not in self.pages
               ), "Attempting to register duplicated ID"
        self.pages[new_page_renderer.id] = new_page_renderer
        if self.current_page == self.NO_PAGES:
            self.current_page = new_page_renderer.id

    def RegisterPages(self, pages: List[Page]) -> None:
        """
            Page order dictated by sequence they were added.
        """
        if not pages:
            return None
        for page in pages:
            self.RegisterPage(page)

    def RenderCurrentPage(self) -> None:
        if self.current_page == self.NO_PAGES:
            st_error(
                "PageManagerError: Attempting to render empty pages sequence.")
        # cache_df = self.context.RestorePageState(self.cache_id)
        self.pages[self.current_page].RenderPage(self.context)
        # self.context.StorePageState(self.cache_df, self.cache_id)

    def GotoPage(self, page_id: str) -> None:
        if page_id not in self.pages:
            raise Exception("Attempting to head to Page {} PageManager {}".format(
                page_id, self.page_manager_id))
        self.current_page = page_id

    def UpdateUser(self) -> None:
        raise NotImplementedError
