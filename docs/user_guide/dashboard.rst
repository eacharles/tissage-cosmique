Dashboard
=========

An interactive Plotly/Dash web interface for exploring stored data.

Launching
---------

::

    pip install "tissage-cosmique[dashboard]"
    tissage-cosmique-dashboard --db-url sqlite+aiosqlite:///my_database.db

Then open http://127.0.0.1:8050 in your browser.

Tabs
----

**Cosmology Parameters**
    Sortable, filterable DataTable of all ``CosmologyParams`` records plus a
    Plotly scatter matrix showing the parameter space coverage.

**Provenance Explorer**
    Summary cards (total executions, success/failure counts, average duration),
    execution duration histogram, status breakdown bar chart. Select an
    execution to inspect its provenance DAG (nodes and edges).

**Emulator Registry**
    DataTable of all ``EmulatorRecord`` entries with training score bar chart
    colored by backend type.

**Codec Registry**
    DataTable of all ``CodecRecord`` entries with codec-type metrics.

Programmatic Access
-------------------

::

    from tissage_cosmique.dashboard import create_dashboard

    app = create_dashboard(db_url="sqlite+aiosqlite:///my.db")
    app.run(host="0.0.0.0", port=8050, debug=True)
