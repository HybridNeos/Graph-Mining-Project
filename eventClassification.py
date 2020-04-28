import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from quickGraph import getTrainTest
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from quickGraph import makeGraph
from sklearn.decomposition import PCA

np.seterr(divide="ignore", invalid="ignore")


def confustionMatrixToGraph(classes, true, predicted, names):
    G = nx.Graph()
    for event in classes:
        G.add_node(event, type="event")

    for athlete in names:
        G.add_node(athlete, type="athlete")

    for name, true, pred in zip(names, y_test, y_pred):
        G.add_edge(name, true)
        G.add_edge(name, pred)

    return G


def testConfusion(clf, X, y):
    y_pred = clf.predict(X)
    score = clf.score(X, y)
    cm = confusion_matrix(y, y_pred)

    # Plot it
    classes = np.append("", max(np.unique(y), np.unique(y_pred), key=lambda x: len(x)))
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    ax.set_title("Binary testing error {:.2f}".format(score))
    plt.show()

    return y_pred, cm


def confusionGraph(true, predicted):
    G = nx.MultiDiGraph()
    for event in max(np.unique(true), np.unique(predicted), key=lambda x: len(x)):
        G.add_node(event)

    for t, p in zip(true, predicted):
        if t != p and (p, t) not in G.edges:
            G.add_edge(t, p)

    return G


def drawConfusionGraph(G):
    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, with_labels=True)
    ax = plt.gca()
    for e in cg.edges:
        ax.annotate(
            "",
            xy=pos[e[0]],
            xycoords="data",
            xytext=pos[e[1]],
            textcoords="data",
            arrowprops=dict(
                arrowstyle="<-",
                color="0.5",
                shrinkA=15,
                shrinkB=15,
                patchA=None,
                patchB=None,
                connectionstyle="arc3,rad=rrr".replace("rrr", str(0.1 * e[2])),
            ),
        )
    for v in G.nodes:
        ax.annotate(v, xy=pos[v] - [0.05, 0.02])
    plt.title("Confusion Graph\nDirection goes true label to predicted") 
    plt.show()

    return None


def shapiro_centrality(events, X, y):
    # Linear model
    le = LabelEncoder().fit(y)
    clf = Ridge().fit(X, le.transform(y))
    return {node: weight for node, weight in zip(events, clf.coef_)}


def centrality_features(G, X, y, num=10):
    centrality = dict()
    centrality["shapiro"] = shapiro_centrality()


if __name__ == "__main__":
    df = pd.read_csv("./fullDetails.csv", index_col=0)
    X_train, y_train, X_test, y_test = getTrainTest(
        df, "RPI", "RIT", True, True, "proportion"
    )

    # Build a classifier
    # clf = KNeighborsClassifier(10).fit(X_train, y_train)
    # clf = SVC(kernel="rbf", gamma="auto").fit(X_train, y_train)
    clf = RidgeClassifier().fit(X_train, y_train)
    # clf = RandomForestClassifier(max_depth=8, random_state=0).fit(X_train, y_train)
    # clf = MultinomialNB().fit(X_train, y_train) # should be method = 3 or 0
    #print(clf.score(X_train, y_train), clf.score(X_test, y_test))

    # get confusion matrix
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Plot it
    classes = np.append(
        "", max(np.unique(y_pred), np.unique(y_test), key=lambda x: len(x))
    )
    fig, ax = plot_confusion_matrix(conf_mat=cm, cmap=plt.cm.coolwarm)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    ax.set_title("RIT binary testing error {:.3f}".format(clf.score(X_test, y_test)))
    plt.show()
    exit()

    # confusion graph
    cg = confusionGraph(y_test, y_pred)
    drawConfusionGraph(cg)
    exit()

    # centrality and weights
    weights = shapiro_centrality(df.columns[5:], X_train, y_train)
    df.loc[:, sorted(weights, key=lambda x: abs(weights[x]), reverse=True)]

    G = makeGraph(df[df.School == "RPI"])
    events = [
        node
        for node, stuff in dict(G.nodes(data=True)).items()
        if stuff["type"] == "event"
    ]
    select = sorted(
        [x[0] for x in nx.degree_centrality(G).items() if x[0] in events],
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    X_train2, y_train2, X_test2, y_test2 = getTrainTest(
        df.loc[:, ["Gender", "School", "State", "Year", "Event"] + select],
        "RPI",
        "RIT",
        True,
        True,
        "feature",
    )
    clf = RidgeClassifier().fit(X_train2, y_train2)
    "{:.3f}, {:.3f}".format(clf.score(X_train2, y_train2), clf.score(X_test2, y_test2))

    #PCA attempt
    pca = PCA(n_components=10)
    X_new = pca.fit_transform(X_train)
    clf = RidgeClassifier().fit(X_new, y_train)
    clf.score(pca.transform(X_test), y_test)
