from .typeforwards import UnifiedContextType
from .cache import Cache, TEXT_T, FLOAT_T

from typing import Dict, List, NewType
import pandas as pd
from streamlit import error as st_error


class Page(object):

    def __init__(self, id: str, page_manager: "PageManager") -> None:
        self.id = id
        self.page_manager = page_manager
        # self.public_cache: CacheType = {}

    def RenderPage(self, ctx: UnifiedContextType) -> None:
        raise NotImplementedError


class PageManager(object):
    __cache_schema = [("ticker", TEXT_T), ("allocation", FLOAT_T)]

    def __init__(self, page_manager_id: str, context: UnifiedContextType):
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
        cache_df = self.context.RestorePageState(self.cache_id)
        self.pages[self.current_page].RenderPage(self.cache_df)
        self.context.StorePageState(self.cache_df, self.cache_id)

    def GotoPage(self, page_id: str) -> None:
        if page_id not in self.pages:
            raise Exception("Attempting to head to Page {} PageManager {}".format(
                page_id, self.page_manager_id))
        self.current_page = page_id

    def UpdateUser(self) -> None:
        raise NotImplementedError
