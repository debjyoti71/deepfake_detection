from graphviz import Digraph

# Create a flowchart diagram
dot = Digraph("AidVerify_Flow", format="png")
dot.attr(rankdir="TB", size="10")

# Styles for different shapes
process = {"shape": "rectangle", "style": "rounded,filled", "fillcolor": "#E6F2FF"}
decision = {"shape": "diamond", "style": "filled", "fillcolor": "#FFF2CC"}
data_store = {"shape": "parallelogram", "style": "filled", "fillcolor": "#E2F7E1"}
terminator = {"shape": "oval", "style": "filled", "fillcolor": "#FFD6D6"}

# Nodes
dot.node("start", "Start", **terminator)

# Donor workflow
dot.node("login_donor", "Login / Signup (Donor)", **process)
dot.node("view_campaigns", "View Active Campaigns", **process)
dot.node("donate_decision", "Donate to a Specific NGO?", **decision)
dot.node("choose_campaign", "Select Campaign", **process)
dot.node("smart_donate", "Smart Donation (ML)", **process)
dot.node("payment", "Make Payment", **process)
dot.node("donation_data", "Donation Data (Blockchain + MongoDB)", **data_store)
dot.node("upload_ss", "Upload Payment Screenshot", **process)
dot.node("donor_done", "Donation Complete", **terminator)

# NGO workflow
dot.node("login_ngo", "Login / Signup (NGO)", **process)
dot.node("submit_campaign", "Submit Campaign Details", **process)
dot.node("docs", "Upload NGO Documents", **process)
dot.node("admin_review", "Admin Review", **process)
dot.node("review_decision", "Approved?", **decision)
dot.node("ngo_dashboard", "NGO Dashboard", **process)
dot.node("bills", "Upload Bills/Invoices", **process)
dot.node("ngo_data", "NGO Campaign Data (Blockchain)", **data_store)

# Field Agent workflow
dot.node("agent_login", "Field Agent Login (Invite)", **process)
dot.node("agent_dashboard", "Agent Dashboard", **process)
dot.node("start_distribution", "Start Distribution", **process)
dot.node("verify_id", "Verify Beneficiary ID + Face", **process)
dot.node("dup_decision", "Already Served?", **decision)
dot.node("record_beneficiary", "Record Beneficiary Data", **process)
dot.node("aid_entry", "Input Distributed Aid", **process)
dot.node("agent_data", "Distribution Data (Blockchain)", **data_store)

# Connections
dot.edges([("start", "login_donor"), ("start", "login_ngo")])

# Donor flow
dot.edge("login_donor", "view_campaigns")
dot.edge("view_campaigns", "donate_decision")
dot.edge("donate_decision", "choose_campaign", label="Yes")
dot.edge("donate_decision", "smart_donate", label="No")
dot.edge("choose_campaign", "payment")
dot.edge("smart_donate", "payment")
dot.edge("payment", "donation_data")
dot.edge("payment", "upload_ss")
dot.edge("upload_ss", "donor_done")

# NGO flow
dot.edge("login_ngo", "submit_campaign")
dot.edge("submit_campaign", "docs")
dot.edge("docs", "admin_review")
dot.edge("admin_review", "review_decision")
dot.edge("review_decision", "ngo_dashboard", label="Yes")
dot.edge("review_decision", "submit_campaign", label="No (Revise)")
dot.edge("ngo_dashboard", "bills")
dot.edge("bills", "ngo_data")

# Field Agent flow
dot.edge("ngo_dashboard", "agent_login")
dot.edge("agent_login", "agent_dashboard")
dot.edge("agent_dashboard", "start_distribution")
dot.edge("start_distribution", "verify_id")
dot.edge("verify_id", "dup_decision")
dot.edge("dup_decision", "record_beneficiary", label="No")
dot.edge("dup_decision", "agent_dashboard", label="Yes")
dot.edge("record_beneficiary", "aid_entry")
dot.edge("aid_entry", "agent_data")

# Save diagram
file_path = "/mnt/data/aidverify_line_flow.png"
dot.render(file_path, cleanup=True)
file_path
