"""Minimal builder for Databricks AI/BI (Lakeview) dashboards.

The labs describe each dashboard as a list of datasets (plain SQL over the gold tables) and widgets,
and this module turns that into the `serialized_dashboard` JSON the Lakeview API expects, then
creates or updates + publishes it. You can always open the result and keep editing it in the UI.
"""

from __future__ import annotations

import json
import uuid


def _id() -> str:
    return uuid.uuid4().hex[:8]


class Dashboard:
    def __init__(self, title: str):
        self.title = title
        self.datasets: list[dict] = []
        self.pages: list[dict] = []
        self._y = 0

    # -- datasets ------------------------------------------------------------------------
    def dataset(self, name: str, sql: str, display_name: str | None = None) -> str:
        self.datasets.append({"name": name, "displayName": display_name or name,
                              "queryLines": [line + "\n" for line in sql.strip().splitlines()]})
        return name

    # -- pages ---------------------------------------------------------------------------
    def page(self, display_name: str):
        self.pages.append({"name": _id(), "displayName": display_name, "layout": [], "pageType": "PAGE_TYPE_CANVAS"})
        self._y = 0
        return self

    def _add(self, widget: dict, width: int, height: int, x: int = 0, new_row: bool = True):
        layout = self.pages[-1]["layout"]
        if not new_row and layout:
            y = layout[-1]["position"]["y"]
        else:
            y = self._y
        layout.append({"widget": widget, "position": {"x": x, "y": y, "width": width, "height": height}})
        self._y = max(self._y, y + height)

    @staticmethod
    def _query(dataset: str, fields: list[tuple[str, str]], disaggregated: bool = False) -> list[dict]:
        return [{"name": "main_query", "query": {"datasetName": dataset, "disaggregated": disaggregated,
                                                 "fields": [{"name": n, "expression": e} for n, e in fields]}}]

    def text(self, markdown: str, width: int = 6, height: int = 2, x: int = 0, new_row: bool = True):
        self._add({"name": _id(), "multilineTextboxSpec": {"lines": [markdown]}}, width, height, x, new_row)

    def counter(self, dataset: str, column: str, title: str, agg: str = "SUM", width: int = 2,
                height: int = 3, x: int = 0, new_row: bool = True):
        expr = f"{agg}(`{column}`)"
        name = f"{agg.lower()}({column})"
        self._add({"name": _id(), "queries": self._query(dataset, [(name, expr)]),
                   "spec": {"version": 2, "widgetType": "counter",
                            "encodings": {"value": {"fieldName": name, "displayName": title}},
                            "frame": {"showTitle": True, "title": title}}}, width, height, x, new_row)

    def chart(self, kind: str, dataset: str, x_col: str, y_col: str, title: str, agg: str = "SUM",
              color: str | None = None, x_scale: str = "categorical", width: int = 3, height: int = 6,
              x: int = 0, new_row: bool = True, x_title: str | None = None, y_title: str | None = None,
              stacked: bool = False):
        """kind: bar | line | area | scatter"""
        y_name = f"{agg.lower()}({y_col})"
        fields = [(x_col, f"`{x_col}`"), (y_name, f"{agg}(`{y_col}`)")]
        enc = {
            "x": {"fieldName": x_col, "scale": {"type": x_scale}, "displayName": x_title or x_col},
            "y": {"fieldName": y_name, "scale": {"type": "quantitative"}, "displayName": y_title or y_col},
        }
        if color:
            fields.append((color, f"`{color}`"))
            enc["color"] = {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color}
        spec = {"version": 3, "widgetType": kind, "encodings": enc, "frame": {"showTitle": True, "title": title}}
        if kind == "bar" and color:
            spec["mark"] = {"layout": "stack" if stacked else "group"}
        self._add({"name": _id(), "queries": self._query(dataset, fields), "spec": spec}, width, height, x, new_row)

    def heatmap(self, dataset: str, x_col: str, y_col: str, value_col: str, title: str, agg: str = "SUM",
                width: int = 3, height: int = 6, x: int = 0, new_row: bool = True):
        v = f"{agg.lower()}({value_col})"
        fields = [(x_col, f"`{x_col}`"), (y_col, f"`{y_col}`"), (v, f"{agg}(`{value_col}`)")]
        spec = {"version": 3, "widgetType": "heatmap", "frame": {"showTitle": True, "title": title},
                "encodings": {"x": {"fieldName": x_col, "scale": {"type": "categorical"}, "displayName": x_col},
                              "y": {"fieldName": y_col, "scale": {"type": "categorical"}, "displayName": y_col},
                              "color": {"fieldName": v, "scale": {"type": "quantitative"}, "displayName": value_col}}}
        self._add({"name": _id(), "queries": self._query(dataset, fields), "spec": spec}, width, height, x, new_row)

    def table(self, dataset: str, columns: list[str], title: str, width: int = 6, height: int = 6,
              x: int = 0, new_row: bool = True):
        fields = [(c, f"`{c}`") for c in columns]
        spec = {"version": 1, "widgetType": "table", "frame": {"showTitle": True, "title": title},
                "encodings": {"columns": [{"fieldName": c, "displayName": c} for c in columns]}}
        self._add({"name": _id(), "queries": self._query(dataset, fields, disaggregated=True), "spec": spec},
                  width, height, x, new_row)

    def filter(self, datasets: list[str], column: str, title: str, width: int = 2, height: int = 2,
               x: int = 0, new_row: bool = True):
        """A multi-select filter applied to every listed dataset that has `column`."""
        queries = [{"name": f"f_{ds}_{column}", "query": {"datasetName": ds, "disaggregated": False,
                                                           "fields": [{"name": column, "expression": f"`{column}`"}]}}
                   for ds in datasets]
        spec = {"version": 2, "widgetType": "filter-multi-select", "frame": {"showTitle": True, "title": title},
                "encodings": {"fields": [{"fieldName": column, "displayName": column, "queryName": q["name"]}
                                         for q in queries]}}
        self._add({"name": _id(), "queries": queries, "spec": spec}, width, height, x, new_row)

    # -- output --------------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({"datasets": self.datasets, "pages": self.pages})

    def publish(self, ws, parent_path: str, warehouse_id: str) -> str:
        """Create (or replace) the dashboard under `parent_path`, publish it, return its URL."""
        from databricks.sdk.service.dashboards import Dashboard as LvDashboard

        existing = None
        try:
            for obj in ws.workspace.list(parent_path):
                if obj.path.endswith(f"/{self.title}.lvdash.json") and getattr(obj, "resource_id", None):
                    existing = obj.resource_id
        except Exception:
            ws.workspace.mkdirs(parent_path)
        body = LvDashboard(display_name=self.title, parent_path=parent_path, warehouse_id=warehouse_id,
                           serialized_dashboard=self.to_json())
        if existing:
            dash = ws.lakeview.update(existing, dashboard=body)
        else:
            dash = ws.lakeview.create(dashboard=body)
        ws.lakeview.publish(dash.dashboard_id, warehouse_id=warehouse_id, embed_credentials=True)
        host = ws.config.host.rstrip("/")
        return f"{host}/dashboardsv3/{dash.dashboard_id}/published"
