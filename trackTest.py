import networkx as nx
import matplotlib.pyplot as plt
from TfrrsHelp import Team, Athlete
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def GendersAndIDs(School, State):
    Men = Team(State, "M", School).getRoster()
    Women = Team(State, "F", School).getRoster()

    Genders = ["M" for _ in range(len(Men))] + ["F" for _ in range(len(Women))]
    IDs = list(Men["Athlete ID"].values) + list(Women["Athlete ID"].values)

    return Genders, IDs

def athleteStats(Genders, IDs, removeEmpty=True):

    # Use ID to make an Athlete class then get relevant information
    def athleteInfo(ID):
        ath = Athlete(ID)
        info = ath.getAthleteInfo()
        return [info["Name"], info["Year"], ath.timesCompetedPerEvent()]

    with ThreadPoolExecutor(max_workers=len(IDs)) as executor:
        result = list(executor.map(athleteInfo, IDs))
    Events = [(result[0], result[1], Genders[i], result[2]) for i in range(len(result))]

    if removeEmpty:
        Events = list(filter(lambda x: len(x[2]) > 0, Events))

    return Events

def toGraph(School, State, Events):
    G = nx.MultiGraph()

    # Create the graph
    for athlete in Events:
        G.add_node(athlete[0], Year=athlete[1], Gender=athlete[2], School=School, State=State)
        for event in athlete[3].keys():
            G.add_edge(athlete[0], event, timesCompeted=athlete[3][event])

    # If empty was kept in, connect them to null event
    if 0 in [d for _, d in G.degree()]:
        G.add_node("None")
        for node in G.nodes(): # impossible to have degree 0 event
            if len(G[node]) == 0:
                g.add_edge(node, "None")

    return G

def drawGraph(G):
    return None

def toMatrix(Names, G):
    df = pd.DataFrame(0, index=Names, columns=set(G.nodes())-set(Names), dtype=int)
    for athlete, event, stats in G.edges.data():
        df.loc[athlete, event] = stats["timesCompeted"] / Years[Names.index(athlete)]
    return df.to_numpy()

#School = "RPI"
#State = "NY"
#Genders, IDs = GendersAndIDs(School, State)
#Events = athleteStats(Genders, IDs)
#G = toGraph(School, State, Events)
#Names = [athlete[0] for athlete in Events]
#X = toMatrix(Names, G)

if __name__ == "__main__":
    State = "NY"
    School = "RIT"
    Men = Team(State, "M", School).getRoster()
    Women = Team(State, "F", School).getRoster()
    IDs = list(Men["Athlete ID"].values) + list(Women["Athlete ID"].values)
    Names = list(Men["NAME"].values) + list(Women["NAME"].values)

    # Attach times down an event to the athlete names
    AthleteStats = []
    Years = []
    with ThreadPoolExecutor(max_workers=len(IDs)) as executor:
        for result in executor.map(Athlete, IDs):
            print(result.getAthleteInfo()["Name"])
            AthleteStats.append(result.timesCompetedPerEvent())
            Years.append(result.getAthleteInfo()["Year"])

    Events = [(name, stats) for name, stats in zip(Names, AthleteStats) if stats]
    Years = [(year, stats) for year, stats in zip(Years, AthleteStats) if stats]

    # Add athlete vertices to the graph
    # make sure RIT order is the same as RPI
    G = nx.Graph()
    for i, name in enumerate(Names):
        Gender = "M" if i < len(Men) else "F"
        G.add_node(name, gender=Gender, School=School, State=State, Year=Years[i])

    # Add the edges
    # use np.where > 0 to turn into simple form
    for name, timesCompeted in Events:
        for event in timesCompeted.keys():
            G.add_edge(name, event, timesCompeted=timesCompeted[event])

    # Deal with people who didn't compete
    drawEmpty = False
    empty = [node for node in G.nodes() if len(G[node]) == 0]
    if drawEmpty:
        G.add_node("None")
        for node in empty:
            G.add_edge(node, "None")
    else:
        for node in empty:
            G.remove_node(node)

    #nx.write_edgelist(G, "trackData.txt", delimiter='_', data=True)

    # Draw it
    def toInitials(name):
        if ", " in name:
            last, first = name.split(", ")
            return first[0] + last[0]
        else:
            return name

    colors = ["Red" if (len(node) < 8 or "(XC)" in node) else "Blue" for node in G.nodes()]
    labels = {node: toInitials(node) for node in G.nodes()}
    nx.draw(G, labels=labels, node_color=colors)
    plt.show()

    # Convert to useful matrix scaled by grade
    df = pd.DataFrame(0, index=Names, columns=set(G.nodes())-set(Names), dtype=int)
    for athlete, event, stats in G.edges.data():
        df.loc[athlete, event] = stats["timesCompeted"]# / Years[Names.index(athlete)]
    df.to_csv("./RIT.data")
    X = df.to_numpy()
