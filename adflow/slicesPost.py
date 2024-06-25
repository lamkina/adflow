# Standard Python modules
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import networkx as nx

# External modules
import numpy as np
from numpy.typing import NDArray


class ADflowSlice:
    def __init__(self) -> None:
        """Datastructure for an ADflow slice."""
        self._data: Dict[str, NDArray] = {}
        self._info: Dict[str, Any] = {}
        self._conn: NDArray = np.array([])

    def create(self, info: Dict[str, Any], conn: NDArray, data: Dict[str, NDArray]) -> None:
        """Method to create the slice. We use an additional method to
        maintain the factory pattern.

        Parameters
        ----------
        info : Dict[str, Any]
            The slice info from the slice header.
        conn : NDArray
            The finite element bar connectivity array.
        data : Dict[str, NDArray]
            A dictionary of variables associated with each coordinate
            in the slice.
        """
        self._info = info
        self._conn = conn
        self._data = data

    @property
    def info(self) -> Dict[str, Any]:
        return self._info

    @property
    def num(self) -> int:
        return self._info["num"]

    @property
    def meshFamily(self) -> str:
        return self._info["family"]

    @property
    def sliceType(self) -> str:
        return self._info["type"]

    @property
    def conn(self) -> NDArray:
        return self._conn

    @property
    def data(self) -> Dict[str, NDArray]:
        return self._data

    @property
    def coordinates(self) -> NDArray:
        return np.column_stack([self._data["CoordinateX"], self._data["CoordinateY"], self._data["CoordinateZ"]])

    def __repr__(self):
        text = f"Slice {self.num:03d}"
        return text

    def __str__(self):
        return f"Slice: num={self._info['num']:03d}, family={self._info['family']}, type={self._info['type']}"


class RegularSlice(ADflowSlice):
    """Regular ADflow planar slice."""

    @property
    def normalAxis(self) -> str:
        return next((axis for axis in ["x", "y", "z"] if axis in self.info), "")

    @property
    def planarCoord(self) -> float:
        axis = self.normalAxis
        return self._info[axis]


class ArbitrarySlice(ADflowSlice):
    """Arbitrary adflow slice."""

    @property
    def point(self) -> NDArray[np.float64]:
        return self._info["point"]

    @property
    def normal(self) -> NDArray[np.float64]:
        return self._info["normal"]


class CylindricalSlice(ADflowSlice):
    """Cylindrical ADflow slice"""

    @property
    def axis(self) -> NDArray[np.float64]:
        return self._info["axis"]

    @property
    def theta(self) -> float:
        return self._info["theta"]


class SliceFactory:
    @staticmethod
    def createSlice(sliceType: str) -> ADflowSlice:
        sliceType = sliceType.lower()
        sl: Optional[ADflowSlice] = None

        if sliceType == "regular":
            sl = RegularSlice()
        elif sliceType == "arbitrary":
            sl = ArbitrarySlice()
        elif sliceType == "cylindrical":
            sl = CylindricalSlice()
        else:
            raise ValueError(f"Invalid slice type: {sliceType}")

        return sl


class SliceReader:
    @staticmethod
    def processSliceHeader(sliceHeader: str) -> Dict:
        """Parses the slice header line using regex.

        Parameters
        ----------
        sliceHeader : str
            The string containing the slice header text.

        Returns
        -------
        Dict
            A dictionary with the info contained in the slice header.

        Raises
        ------
        ValueError
            Raised if the slice number is not found in the header.
        ValueError
            Raised if the mesh family is not found in the header.
        ValueError
            Raised if the slice type is not found in the header.
        ValueError
            Raised if the point or normal vector is not found for an
            arbitrary or cylindrical slice type.
        ValueError
            Raised if the axis or theta is not found for a cylindrical
            slice type.
        """
        # We want to pull all the information out of the slice header
        # Regular expression pattern for slice number
        sliceNumberPattern = r"Slice_(\d+)"  # Get the slice number
        sliceFamilyPattern = r"Slice_\d+\s+(\w+)"  # Get the slice family
        sliceTypePattern = r"\b(Relative|Absolute)\s+(\w+)"  # Get the slice type: Regular, Arbitrary, Cylindrical

        # Extract the slice number
        sliceNumberMatch = re.search(sliceNumberPattern, sliceHeader)
        if sliceNumberMatch is None:
            raise ValueError("Could not find slice number in the slice header.")

        sliceFamilyMatch = re.search(sliceFamilyPattern, sliceHeader)
        if sliceFamilyMatch is None:
            raise ValueError("Could not find slice family in the slice header.")

        sliceTypeMatch = re.search(sliceTypePattern, sliceHeader)
        if sliceTypeMatch is None:
            raise ValueError("Could not find slice type in the slice header.")

        sliceNumber = sliceNumberMatch.group(1)
        sliceFamily = sliceFamilyMatch.group(1)
        sliceType = sliceTypeMatch.group(2)

        sliceInfo = {"num": int(sliceNumber), "family": sliceFamily, "type": sliceType}

        if sliceType.lower() == "arbitrary":
            point_pattern = (
                r"Point\s*=\s*\(([\d\.-]+),\s*([\d\.-]+),\s*([\d\.-]+)\)"  # Get the point if slice type is Arb or Cylin
            )
            normalPattern = r"Normal\s*=\s*\(([\d\.-]+),\s*([\d\.-]+),\s*([\d\.-]+)\)"  # Get the normal vector if slice type is Arb or Cylin
            pointMatch = re.search(point_pattern, sliceHeader)
            normalMatch = re.search(normalPattern, sliceHeader)

            if pointMatch is None or normalMatch is None:
                raise ValueError("Could not find point or normal in the slice header.")

            point = np.array(pointMatch.group(1, 2, 3), dtype=float)
            normal = np.array(normalMatch.group(1, 2, 3), dtype=float)
            sliceInfo["point"] = point
            sliceInfo["normal"] = normal

        elif sliceType.lower() == "cylindrical":
            axisPattern = (
                r"Axis\s*=\s*\(([\d\.-]+),\s*([\d\.-]+),\s*([\d\.-]+)\)"  # Get the axis if slice type is Cylin
            )
            thetaPattern = r"Theta\s*=\s*([\d.]+)"  # Get theta if the slice type is cylin

            axisMatch = re.search(axisPattern, sliceHeader)
            thetaMatch = re.search(thetaPattern, sliceHeader)

            if axisMatch is None or thetaMatch is None:
                raise ValueError("Could not find axis or theta in the slice header.")

            axis = np.array(axisMatch.group(1, 2, 3), dtype=float)
            theta = float(thetaMatch.group(1))
            sliceInfo["axis"] = axis
            sliceInfo["theta"] = theta

        elif sliceType.lower() == "regular":
            coordValues = {}
            for coord in ["x", "y", "z"]:
                match = re.search(rf"Regular\s+{coord}\s*=\s*([\d.-]+)", sliceHeader)
                if match:
                    coordValues[coord] = float(match.group(1))

            sliceInfo.update(coordValues)

        return sliceInfo

    @staticmethod
    def _readSlicefile(filename: Path) -> Iterator[ADflowSlice]:
        """Loops over a slice file, reads the contents line by line, and
        creates the requisite ADflow slice based on the slice type.

        Parameters
        ----------
        filename : Path
            The path to the slice file.

        Yields
        ------
        Iterator[ADflowSlice]
            An iterator of ADflow slices.
        """
        with open(filename, "r") as f:
            lines = f.readlines()

        # first line is the title
        # second line has the variable list
        varPattern = r'"([^"]+)"'
        varList = re.findall(varPattern, lines[1])

        # number of variables
        nVar = len(varList)

        # beginning line index for the current slice
        # this is initialized to 2 for the first slice
        sliceBegin = 2
        islice = 0

        # loop over slices
        while sliceBegin < len(lines):
            sliceHeader = lines[sliceBegin]
            sliceCounts = lines[sliceBegin + 1].replace("=", "").split()
            nNode = int(sliceCounts[1])
            nElem = int(sliceCounts[3])

            if "Zone T" in sliceHeader:  # Make sure we have a slice header
                # Get the slice info from the header as a dict
                sliceInfo = SliceReader.processSliceHeader(sliceHeader)

                # indices where the node data and conn starts
                dataBeg = sliceBegin + 3
                connBeg = dataBeg + nNode

                # load the node data
                nodeData = np.genfromtxt(
                    filename,
                    dtype={
                        "names": varList,
                        "formats": ["f4"] * nVar,
                    },
                    skip_header=dataBeg,
                    max_rows=nNode,
                )

                # Put the node data into dictionary
                sliceData = {var: nodeData[var] for var in varList}

                # load the connectivity
                conn1, conn2 = np.genfromtxt(
                    filename,
                    dtype={
                        "names": (
                            "conn1",
                            "conn2",
                        ),
                        "formats": ("i4", "i4"),
                    },
                    usecols=(0, 1),
                    skip_header=connBeg,
                    max_rows=nElem,
                    unpack=True,
                    invalid_raise=False,
                )

                # -1 is to get back to numpy indexing
                conn = np.column_stack([conn1, conn2]) - 1

                sl = SliceFactory.createSlice(sliceInfo["type"])
                sl.create(sliceInfo, conn, sliceData)
                yield sl

                islice += 1

            # Increment the beginning line index for the next slice
            # + 1 is for the Datapacking=point line, + 2 is for the 2 headers
            sliceBegin += 1 + nNode + nElem + 2


class SliceBuilder:
    def __init__(self) -> None:
        self._slices: List[ADflowSlice] = []
        self._filename: Path = Path("")
        self._range: Tuple[int | None, int | None] = (None, None)

    @property
    def slices(self) -> List[ADflowSlice]:
        return self._slices

    @property
    def filename(self) -> Path:
        return self._filename

    def setup(self, filename: Union[str, Path], sliceRange: slice) -> None:
        self._filename = Path(filename)
        self._range = sliceRange

    def buildSlices(self) -> None:
        if not self._filename.exists():
            raise FileNotFoundError(f"File {self._filename} does not exist.")

        reader = SliceReader()
        self._slices = list(reader._readSlicefile(self._filename))[self._range]


@dataclass
class FiniteElementCurve:
    """Finite element curve class.

    Attributes
    ----------
    coordinates: NDArray[np.float64]
        The sorted coordinates of the curve.
    data: Dict[str, NDArray]
        The sorted discrete data associated with each coordinate.
    conn: NDArray[np.int64]
        The finite element bar-element connectivity array.
    """

    coordinates: NDArray[np.float64]
    data: Dict[str, NDArray]
    conn: NDArray[np.int64]
    info: Dict[str, Any]


def _removeZeroLengthEdges(coords: NDArray[np.float64], conn: NDArray[np.int64]) -> NDArray[np.int64]:
    """Remove zero length edges from the connectivity array.

    Parameters
    ----------
    coords : NDArray[np.float64]
        The coordinates of the nodes.
    conn : NDArray[np.int64]
        The connectivity array.

    Returns
    -------
    NDArray[np.int64]
        The connectivity array with zero length edges removed.
    """
    mask = np.linalg.norm(coords[conn[:, 0]] - coords[conn[:, 1]], axis=1) > 1e-12
    return conn[mask]


def _buildGraph(coords: NDArray[np.float64], conn: NDArray[np.int64]) -> nx.Graph:
    """Build a graph from the connectivity array.

    There's 4 different combinations of connectivity between bar elements.
    We need to brute force check all of the possibilities to build the connectivity graph.

    1. Start of one element connects to the end of another element (s2e)
    2. End of one element connects to the end of another element (e2e)
    3. Start of one element connects to the start of another element (s2s)
    4. End of one element connects to the start of another element (e2s)

    Parameters
    ----------
    coords : NDArray[np.float64]
        The coordinates of the nodes.
    conn : NDArray[np.int64]
        The connectivity array.

    Returns
    -------
    nx.Graph
        The graph representation of the connectivity array.
    """
    startCoords, endCoords = coords[conn[:, 0]], coords[conn[:, 1]]
    G = nx.Graph()
    G.add_nodes_from(map(tuple, conn))
    nConn = len(conn)

    # Compute the pairwise distances for all combinations, using broadcasting and avoiding redundant stacking.
    distMtxS2E = np.linalg.norm(startCoords[:, None, :] - endCoords, axis=2)
    distMtxE2E = np.linalg.norm(endCoords[:, None, :] - endCoords, axis=2)
    distMtxS2S = np.linalg.norm(startCoords[:, None, :] - startCoords, axis=2)
    distMtxE2S = np.linalg.norm(endCoords[:, None, :] - startCoords, axis=2)

    # Threshold value to consider distances as zero
    threshold = 1e-12

    # Find indices where the distances are below the threshold but are not the diagonal (self-connections)
    edgesS2E = np.where((distMtxS2E < threshold) & (np.arange(nConn)[:, None] != np.arange(nConn)))
    edgesE2E = np.where((distMtxE2E < threshold) & (np.arange(nConn)[:, None] != np.arange(nConn)))
    edgesS2S = np.where((distMtxS2S < threshold) & (np.arange(nConn)[:, None] != np.arange(nConn)))
    edgesE2S = np.where((distMtxE2S < threshold) & (np.arange(nConn)[:, None] != np.arange(nConn)))

    # Concatenate all edges and using numpy's advanced indexing to map to the original connections
    allEdges = np.concatenate(
        (
            conn[edgesS2E[0], :],
            conn[edgesE2E[0], :],
            conn[edgesS2S[0], :],
            conn[edgesE2S[0], :],
        ),
        axis=0,
    )
    allEdgePairs = np.concatenate(
        (
            conn[edgesS2E[1], :],
            conn[edgesE2E[1], :],
            conn[edgesS2S[1], :],
            conn[edgesE2S[1], :],
        ),
        axis=0,
    )

    # Convert the edge nodes to tuples (if they aren't already) and add them to the graph
    edgesToAdd = [(tuple(edgeStart), tuple(edgeEnd)) for edgeStart, edgeEnd in zip(allEdges, allEdgePairs)]
    G.add_edges_from(edgesToAdd)

    return G


def _findCurveConns(G: nx.Graph) -> List[NDArray[np.int64]]:
    """Find the connected curves in the graph and sort them using a
    depth first search.

    Parameters
    ----------
    G : nx.Graph
        The graph representation of the connectivity array.

    Returns
    -------
    List[NDArray[np.int64]]
        A list of the sorted connectivity arrays for each curve.

    Raises
    ------
    ValueError
        If the number of end points is not 0 or 2 (for closed and
        open curves, respectively).
    """
    # Get the curves based on graph connections
    curves = list(nx.connected_components(G))

    # Sort the curves using a depth first preorder node search
    sortedCurves = []
    for curve in curves:
        endPoints = [node for node in curve if G.degree(node) == 1]

        if len(endPoints) == 2:
            # We have an open curve, find the node with the smallest starting index
            startingPoint = min(endPoints, key=lambda x: x[0])

        elif len(endPoints) == 0:
            # We have a closed curve, pick the first starting point
            startingPoint = next(iter(curve))

        else:
            raise ValueError(
                f"Found {len(endPoints)} number of end points.\n"
                "Valid values are 2 end points for an open curve and 0 end points for a closed curve."
            )

        sortedCurve = list(nx.dfs_preorder_nodes(G, startingPoint))

        if not endPoints:
            # Make sure the curve is closed if it's a closed curve
            sortedCurve.append(startingPoint)

        sortedCurves.append(sortedCurve)

    return [np.row_stack(curve) for curve in sortedCurves]


def _fixOpenCurve(coords: NDArray[np.float64], conn: NDArray[np.int64]) -> NDArray[np.int64]:
    """Fix the orientation of an open curve so that the end of one
    element connects to the start of another element.

    Thinking of each line segment as a vector, the orientation needs to
    be such that the end of A connects to the start of B. If A and B
    each have two elements (A0, A1) and (B0, B1), there are three bad
    orientations we need to correct:

    1. A1---A0 = B1---B0 (reverse A and B)
    2. A1---A0 = B0---B1 (reverse A)
    3. A0---A1 = B1---B0 (reverse B)

    Parameters
    ----------
    coords : NDArray[np.float64]
        The coordinates of the nodes.
    conn : NDArray[np.int64]
        The connectivity array.

    Returns
    -------
    NDArray[np.int64]
        The corrected connectivity array.
    """
    startA = coords[conn[0]]
    startB = coords[conn[1]]
    endA = coords[conn[-2]]
    endB = coords[conn[-1]]

    if np.array_equal(startA[0], startB[-1]):
        conn[0] = conn[0, ::-1]
        conn[1] = conn[1, ::-1]

    elif np.array_equal(startA[0], startB[0]):
        conn[0] = conn[0, ::-1]

    elif np.array_equal(startA[-1], startB[-1]):
        conn[1] = conn[1, ::-1]

    if np.array_equal(endA[0], endB[-1]):
        conn[-2] = conn[-2, ::-1]
        conn[-1] = conn[-1, ::-1]

    elif np.array_equal(endA[0], endB[0]):
        conn[-2] = conn[-2, ::-1]

    elif np.array_equal(endA[-1], endB[-1]):
        conn[-1] = conn[-1, ::-1]

    return conn


def feSort(coords: NDArray[np.floating], conn: NDArray[np.int64]) -> List[NDArray[np.int64]]:
    """Sort the connectivity array into a list of curves.

    Parameters
    ----------
    coords : NDArray[np.floating]
        The coordinates of the nodes.
    conn : NDArray[np.int64]
        The connectivity array.

    Returns
    -------
    List[NDArray[np.int64]]
        A list of sorted connectivities, each representing a curve.
    """
    conn = _removeZeroLengthEdges(coords, conn)

    G = _buildGraph(coords, conn)

    curveConns = _findCurveConns(G)

    sortedCurveConns = []
    for curveConn in curveConns:
        newConn = curveConn.copy()
        newConn = _fixOpenCurve(coords, newConn)
        sortedCurveConns.append(newConn[:, 0])

    return sortedCurveConns


class FiniteElementCurveBuilder:
    def __init__(self) -> None:
        self._coords: NDArray[np.float64] = np.array([])
        self._data: Dict[str, NDArray] = {}
        self._conn: NDArray[np.int64] = np.array([])
        self._info: Dict[str, Any] = {}

    def setup(self, sl: ADflowSlice) -> None:
        self._coords = sl.coordinates
        self._data = sl.data
        self._conn = sl.conn
        self._info = sl.info

    def build(self) -> List[FiniteElementCurve]:
        if self._coords.size == 0:
            raise ValueError("Coordinates must be set.")
        if not self._data:
            raise ValueError("Data must be set.")
        if self._conn.size == 0:
            raise ValueError("Connectivity must be set.")

        sortedCurveConns = feSort(self._coords, self._conn)
        curves = []
        for curveConn in sortedCurveConns:
            coords = self._coords[curveConn]
            data = {key: value[curveConn] for key, value in self._data.items()}
            curves.append(FiniteElementCurve(coords, data, curveConn))

        return curves


class SliceInterface:
    def __init__(self, sliceFile: Union[str, Path]) -> None:
        """Creates an interface to build slices and finite element curves.

        Parameters
        ----------
        sliceFile : Union[str, Path]
            A pathlike object for the slice file from ADflow.
        """
        self._file = sliceFile
        self._sliceBuilder = SliceBuilder()
        self._feCurveBuilder = FiniteElementCurveBuilder()

    def _createSlices(self, sliceRange: slice) -> List[ADflowSlice]:
        """Utility method to setup and build slices.

        Parameters
        ----------
        sliceRange : slice
            The range of slices to be returned. This uses the default
            python 'slice' object.

        Returns
        -------
        List[ADflowSlice]
            A list of ADflow slices.
        """
        self._sliceBuilder.setup(self._file, sliceRange=sliceRange)
        self._sliceBuilder.buildSlices()

    def extractCurves(self, sliceRange: Optional[slice] = None) -> List[FiniteElementCurve]:
        """Creates the slices and finite element curves from the slice
        file.

        Parameters
        ----------
        sliceRange : Optional[slice], optional
            The range of slices to use for building the curves. This
            uses the default python 'slice' object, by default None

        Returns
        -------
        List[FiniteElementCurve]
            A list of sorted finite element curves.
        """
        if sliceRange is None:
            sliceRange = slice(None, None)

        self._createSlices(sliceRange)

        feCurves = []
        for sl in self._sliceBuilder.slices:
            self._feCurveBuilder.setup(sl)
            localCurves = self._feCurveBuilder.build()
            feCurves.extend(localCurves)

        return feCurves

    def extractSlices(self, sliceRange: Optional[slice] = None) -> List[ADflowSlice]:
        """Reads the slice file and creates the unordered slice data,
        returned as an ADflow slice.

        Parameters
        ----------
        sliceRange : Optional[slice], optional
            The range to use for building the slices. This uses the
            default python 'slice' object, by default None

        Returns
        -------
        List[ADflowSlice]
            A list of ADflow slices.
        """
        if sliceRange is None:
            sliceRange = slice(None, None)

        self._createSlices(sliceRange)

        return self._sliceBuilder.slices
