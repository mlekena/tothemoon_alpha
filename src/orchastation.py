from .cache import Cache, TEXT_T, FLOAT_T

import pandas as pd
from typing import Dict, List, Optional, Any
from streamlit import error as st_error
from sqlalchemy import Table, MetaData, Column, String, insert, select, update, func
from sqlalchemy.engine import ResultProxy


class UnifiedContext(object):
    """
    `current_page` is the current page to render given the correct page manager.
    If request from the wrong page manager, an assertion is raised.
    """
    # Passes Cache into Page render calls
    cache_schema = [("currentPage", TEXT_T)]
    # "_id_unified_context_cache"
    UNIFIED_CONTEXT_CACHE_ID = "_id_unified_context_cache_x1"

    """ Cache Data Helper Functions """
    @staticmethod
    def __MakeTable(id: str, metadata: MetaData) -> Table:
        return Table(id, metadata,
                     Column("user_id", String(320), primary_key=True),
                     Column("current_page", String(255)))

    def _insert_uc_cache_statement(self) -> Any:
        return insert(self.table).values(user_id=self.user, current_page=self.current_page)

    def _UnifiedContextCacheIsEmpty(self) -> bool:
        result = self.cache.connection.execute(
            select(func.count(self.table.c.user_id)).where(self.table.c.user_id == self.user))
        empty: bool = result.scalar() == 0
        return empty

    def _SelectUnifiedContextDataStatement(self) -> Any:
        return select(self.table).where(self.table.c.user_id == self.user)

    def _UpdateUnifiedCOntextDataStatement(self) -> Any:
        return update(self.table) \
            .where(self.table.c.user_id == self.user) \
            .values(current_page=(self.current_page))

    def __init__(self, user: str) -> None:
        self.user = user
        self.cache = Cache.get_instance(Cache.default_connection_url)
        self.table: Table = self.cache.RegisterCacheTable(UnifiedContext.UNIFIED_CONTEXT_CACHE_ID,
                                                          UnifiedContext.__MakeTable)
        self.cache.CreateRegisteredTablesAndConnect()
        if self.table == None:
            raise RuntimeError("Failed to create or get table.")
        if self._UnifiedContextCacheIsEmpty():
            # Insert defaults for this user.
            print("Creating user ${self.user}context")
            self.current_page: str = ""
            self.cache.CacheInsert(self._insert_uc_cache_statement())
        else:
            # load in this users context data.
            print("Retrieving user ${self.user} cached data")
            retrieved_data = self.cache.CacheDataSelect(
                self._SelectUnifiedContextDataStatement())
            self.current_page = retrieved_data.current_page

        # self.page_managers: Dict[str, PageManager] = {}
        self.user_data: pd.DataFrame = pd.DataFrame([])
        self.unified_context_id = "_id_UnifiedContextCoreCache"
        self.cache.InitCache(self.unified_context_id,
                             UnifiedContext.cache_schema)

        # self.app_state: pd.DataFrame = self.cache.read_state_df(
        #     self.unified_context_id)
    # String type used to get forward usage of type names
    # https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class

    def SetCurrentPage(self, page_id: str) -> None:
        # TODO Think about how to ensure this gets called before each render
        # if page_id not in self.page_managers:
        #     raise Exception(
        #         "Setting to unregistered PageManager ${page_id}")
        self.current_page = page_id

    def GetCurrentPage(self) -> str:
        return self.current_page

    def RestorePageState(self, page_manager_id: str) -> pd.DataFrame:
        print("Restore State")
        return self.cache.read_state_df(page_manager_id)

    def StorePageState(self, state: pd.DataFrame, page_manager_id: str) -> None:
        print("Store State")
        self.cache.write_state_df(state, page_manager_id)

    def StoreContextAndClear(self) -> None:
        self.cache.CacheDataUpdate(self._UpdateUnifiedCOntextDataStatement())
        self.cache.ClearCacheRegistry()


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


def StoreUnifiedContext(ctx: UnifiedContext) -> None:
    ctx.StoreContextAndClear()


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
        # self.current_page: str = self.NO_PAGES
        self.cache_df: pd.DataFrame = None
        self.context = context
        # context.InitCache(self.cache_id, PageManager.cache_schema)

    def RegisterPage(self, new_page_renderer: Page) -> None:
        assert(new_page_renderer.id not in self.pages
               ), "Attempting to register duplicated ID"
        if len(self.pages) == 0:
            self.default_page_id = new_page_renderer.id
        self.pages[new_page_renderer.id] = new_page_renderer
        # if self.current_page == self.NO_PAGES:
        # self.current_page = self.default_page_id

    def RegisterPages(self, pages: List[Page]) -> None:
        """
            Page order dictated by sequence they were added.
        """
        if not pages:
            return None
        for page in pages:
            self.RegisterPage(page)

    def RenderCurrentPage(self) -> None:
        # if self.current_page == self.NO_PAGES:
        #     st_error(
        #         "PageManagerError: Attempting to render empty pages sequence.")
        if self.context.current_page not in self.pages:
            self.context.SetCurrentPage(self.default_page_id)
        # cache_df = self.context.RestorePageState(self.cache_id)
        self.pages[self.context.current_page].RenderPage(self.context)
        # self.context.StorePageState(self.cache_df, self.cache_id)

    def GotoPage(self, page_id: str) -> None:
        if page_id not in self.pages:
            raise Exception("Attempting to head to Page {} PageManager {}".format(
                page_id, self.page_manager_id))
        self.context.SetCurrentPage(page_id)

    def UpdateUser(self) -> None:
        raise NotImplementedError
