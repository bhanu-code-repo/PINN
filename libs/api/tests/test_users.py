"""Tests for the UserManager (bcrypt-backed user management)."""

import pytest
from api.users import UserManager


@pytest.fixture
def mgr(tmp_path):
    with UserManager(tmp_path / "users.db") as m:
        yield m


class TestCreateUser:
    def test_create_and_get(self, mgr):
        user = mgr.create_user("alice", "secret123")
        assert user.username == "alice"
        assert user.is_admin is False
        assert user.groups == []

    def test_create_with_groups_and_admin(self, mgr):
        user = mgr.create_user("bob", "pw", groups=["research", "eng"], is_admin=True)
        assert user.is_admin is True
        assert set(user.groups) == {"research", "eng"}

    def test_duplicate_raises(self, mgr):
        mgr.create_user("alice", "pw")
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_user("alice", "pw2")


class TestAuthenticate:
    def test_valid_credentials(self, mgr):
        mgr.create_user("alice", "correct")
        user = mgr.authenticate("alice", "correct")
        assert user is not None
        assert user.username == "alice"

    def test_wrong_password(self, mgr):
        mgr.create_user("alice", "correct")
        assert mgr.authenticate("alice", "wrong") is None

    def test_unknown_user(self, mgr):
        assert mgr.authenticate("ghost", "pw") is None


class TestPasswordHashing:
    def test_passwords_are_hashed(self, mgr):
        mgr.create_user("alice", "plaintext")
        row = mgr._conn.execute(
            "SELECT password FROM users WHERE username = ?", ("alice",)
        ).fetchone()
        stored = row[0]
        assert stored != "plaintext"
        assert stored.startswith("$2")  # bcrypt prefix

    def test_verify_password(self):
        hashed = UserManager.hash_password("test")
        assert UserManager.verify_password("test", hashed)
        assert not UserManager.verify_password("wrong", hashed)


class TestUpdatePassword:
    def test_update_and_auth(self, mgr):
        mgr.create_user("alice", "old")
        mgr.update_password("alice", "new")
        assert mgr.authenticate("alice", "old") is None
        assert mgr.authenticate("alice", "new") is not None

    def test_update_nonexistent_raises(self, mgr):
        with pytest.raises(KeyError):
            mgr.update_password("ghost", "pw")


class TestUpdateUser:
    def test_update_groups(self, mgr):
        mgr.create_user("alice", "pw", groups=["a"])
        updated = mgr.update_user("alice", groups=["b", "c"])
        assert set(updated.groups) == {"b", "c"}

    def test_promote_to_admin(self, mgr):
        mgr.create_user("alice", "pw")
        updated = mgr.update_user("alice", is_admin=True)
        assert updated.is_admin is True

    def test_demote_admin(self, mgr):
        mgr.create_user("alice", "pw", is_admin=True)
        updated = mgr.update_user("alice", is_admin=False)
        assert updated.is_admin is False


class TestDeleteUser:
    def test_delete_and_verify(self, mgr):
        mgr.create_user("alice", "pw")
        mgr.delete_user("alice")
        with pytest.raises(KeyError):
            mgr.get_user("alice")

    def test_delete_nonexistent_raises(self, mgr):
        with pytest.raises(KeyError):
            mgr.delete_user("ghost")


class TestListAndCount:
    def test_empty(self, mgr):
        assert mgr.list_users() == []
        assert mgr.count() == 0

    def test_multiple_users(self, mgr):
        mgr.create_user("alice", "pw")
        mgr.create_user("bob", "pw")
        assert mgr.count() == 2
        names = [u.username for u in mgr.list_users()]
        assert names == ["alice", "bob"]  # ORDER BY username


class TestEnsureAdmin:
    def test_creates_default_admin(self, mgr):
        mgr.ensure_admin()
        admin = mgr.get_user("admin")
        assert admin.is_admin is True
        assert "admin" in admin.groups

    def test_promotes_existing_non_admin(self, mgr):
        mgr.create_user("admin", "pw", is_admin=False)
        mgr.ensure_admin()
        admin = mgr.get_user("admin")
        assert admin.is_admin is True

    def test_noop_if_admin_exists(self, mgr):
        mgr.create_user("boss", "pw", is_admin=True)
        mgr.ensure_admin()
        # Default admin should not be created
        assert mgr.count() == 1
