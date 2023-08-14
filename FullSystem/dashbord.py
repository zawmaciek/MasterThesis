import dash_bootstrap_components as dbc
from system import System
from dataset import movieId
from dash import Dash, html, dcc, callback, Output, Input, dash_table

s = System()
app = Dash(external_stylesheets=[dbc.themes.LUX])
app.layout = html.Div([
    html.H1(children='Movie Recommender System', style={'textAlign': 'center'}),
    html.P(children="Select movies using input below"),
    dcc.Dropdown(
        options=s.dataset.movie_id_title_mapping,
        multi=True,
        id='movie-selection'
    ),
    html.P(children="Results:"),
    html.P(id="recommendations", children="Select any movies to see recommendations")
])


@callback(
    Output('recommendations', 'children'),
    Input('movie-selection', 'value')
)
def update_graph(value):
    if value not in [None, []]:
        ratings = [(movieId(int(a)), 1.0) for a in value]
        print(ratings)
        df, ensemble_df = s.get_recommendations_for_user(ratings)
        return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]), dash_table.DataTable(
            ensemble_df.to_dict('records'), [{"name": i, "id": i} for i in ensemble_df.columns])
    else:
        return "Select any movies to see recommendations"


if __name__ == '__main__':
    app.run(debug=False)
