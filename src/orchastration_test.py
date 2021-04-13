from . orchastation import UnifiedContext

test_user = "test_user_id"


class TestUnifiedContext:
    def setup_method(self) -> None:
        self.context = UnifiedContext(test_user)
    # @classmethod

    def teardown_method(self) -> None:
        self.muc.connection.close()  # type: ignore

    def test_new_context_is_not_empty(self) -> None:
        ctx = UnifiedContext("new_user")
        ctx
