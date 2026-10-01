"""
LakeRCM Agent — User identity resolution.

Maps authenticated email addresses to display names and user context.
"""


def resolve_user_identity(email: str) -> dict:
    """Resolve a user email to display name and context.

    Returns: {"email": str, "display_name": str, "first_name": str, "role": str}
    """
    # Derive display name from email
    local = email.split("@")[0] if "@" in email else email
    parts = local.replace(".", " ").replace("_", " ").replace("-", " ").title().split()
    first_name = parts[0] if parts else "User"
    display_name = " ".join(parts) if parts else email

    return {
        "email": email,
        "display_name": display_name,
        "first_name": first_name,
        "role": "Reviewer",
    }


def get_user_for_email(email: str) -> dict:
    """Get user context for the agent, given an authenticated email."""
    identity = resolve_user_identity(email)
    return {
        "user_email": identity["email"],
        "display_name": identity["display_name"],
        "first_name": identity["first_name"],
        "role": identity["role"],
    }
