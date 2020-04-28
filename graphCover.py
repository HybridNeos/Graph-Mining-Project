def setCheck(candidateEvents, uniqueMaximals):
    # If something in uniqueMaximals is a supserset of candidatEvents we don't
    #  candidatEvents
    # If something in uniqueMaximals is a subset of candidatEvents we add
    #   candidatEvents and remove it
    supersets = set()
    subsets = set()

    for athlete, events in uniqueMaximals.items():
        if events.issuperset(candidateEvents):
            supersets.add(athlete)
            break
        # Since issuperset includes equality we know gives only proper supersets
        elif events.issubset(candidateEvents):
            subsets.add(athlete)

    return supersets, subsets


def getUniqueMaximals(subsets):
    uniqueMaximals = {}
    for athlete, events in subsets.items():
        # people in uniqueMaximals who are subsets/supersets of events
        supersets, subsets = setCheck(events, uniqueMaximals)
        if not supersets:
            uniqueMaximals[athlete] = events
            # remove any subsets in uniqueMaximals from adding the new events
            for noLongerMaximal in subsets:
                del uniqueMaximals[noLongerMaximal]

    return uniqueMaximals


def subGraph(subsets):
    H = nx.Graph()

    for athlete in subsets.keys():
        H.add_node(athlete, type="athlete")

    for event in reduce(lambda x, y: x.union(y), subsets.values()):
        H.add_node(event, type="event")

    for athlete in subsets.keys():
        for event in subsets[athlete]:
            H.add_edge(athlete, event)

    return H


# taken from martinbroadhurst.com/greedy-set-cover-in-python.html
def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)

    # Check the subsets cover the universe
    if elements.issubset(universe) and elements != universe:
        return None

    covered = set()
    cover = []

    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        covered |= subset

    return cover


def removeMultiEvents(subsets, G):
    newSubsets = deepcopy(subsets)

    for athlete, events in subsets.items():
        if "Pent" in events:
            newSubsets[athlete] -= set(["HJ", "SP", "LJ", "60H", "800"])
        if "Hep" in events:
            if G.nodes(data=True)[athlete]["Gender"] == "F":
                newSubsets[athlete] -= set(["100H", "HJ", "SP", "200", "LJ", "JT", "800"])
            else:
                newSubsets[athlete] -= set(["1000", "60", "LJ", "SP", "HJ", "60H", "PV"])
        if "Dec" in events:
            newSubsets[athlete] -= set(["100", "LJ", "SP", "HJ", "400", "110H", "DT", "PV", "JT", "1500"])

    return newSubsets

"""def optimizeFurther(subsets, G):
    # find the degree 1 events, who does them, and those people's events
    lonelyEvents = [node for node, degree in G.degree() if degree == 1]
    H = G.copy()

    # find the people
    trivialPeople = set()
    for event in lonelyEvents:
        trivialPeople |= set(H[event])

    # find the events
    trivialEvents = set()
    for person in trivialPeople:
        trivialEvents |= set(H[person])

    # remove them
    for node in trivialPeople.union(trivialEvents):
        H.remove_node(n

    return {node: set(H[node]) for node in H.nodes() if node in subsets.keys()}, trivialPeople, trivialEvents

### MAKES GREEDY WORSE
# Remove the 2-hop neighborhood of degree 1 vertices
    if False:
    reducedMaxAthletes, trivialPeople, trivialEvents = optimizeFurther(maxAthletes, H)
    maxAthletes = reducedMaxAthletes
    universe -= trivialEvents"""
