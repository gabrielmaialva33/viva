defmodule VivaWeb.IndexLiveTest do
  use VivaWeb.ConnCase, async: true
  import Phoenix.LiveViewTest

  test "disconnected and connected render", %{conn: conn} do
    {:ok, page_live, disconnected_html} = live(conn, "/")
    assert disconnected_html =~ "VIVA"
    assert render(page_live) =~ "AI Avatar Platform"
  end
end
