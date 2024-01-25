import numpy as np
import uuid


class HalfEdgeObject:    
    def __init__(self, faces) -> None:
        self.__faces=faces
        self.__edges=[]
        self.__vertsDict={}
        self.__edgeDict={}
        self.__faceDict={}
        self.__halfEdges={}

        for i in range(len(faces)):
            self.__bindFace(i)

    def getEdgesOfVert(self,vert):
        hes = self.__vertsDict.get(vert,[])
        edges=[]
        for he in hes:
            info=self.__getHalfEdgeInfo(he)
            edges.append([info['vert_from'],info['vert_to']])
        return edges
    
    def getFaceofEdges(self,edges):
        faceIndices=set()
        for e in edges:
            edgeIndex = self.__getEdgeByVerts(e)
            faceIndices=faceIndices.intersection(self.__getFacesOfEdge(edgeIndex))
        if len(faceIndices)==0:
            return None
        assert len(faceIndices)==1
        return self.__getFace(faceIndices[0])

    def __getFacesOfEdge(self, edgeIndex):
        hes = self.__edgeDict[edgeIndex]
        return [self.__getHalfEdgeInfo(he)['face'] for he in hes]

    def __bindFace(self, faceIndex):
        face = self.__getFace(faceIndex)
        hes=[]
        for i,vert in enumerate(face):
            hes.append(self.__bindEdge(face[i-1],vert))
            if len(hes)>1:
                self.__heFromTo(hes[-2],hes[-1])
        for he in hes:
            self.__heAddFace(he,faceIndex)
        return hes


    def __bindEdge(self, v1, v2):
        edgeIndex=self.__getOrAddEdge(v1, v2)
        he=self.__bindVert(v1,v2)
        self.__heAddEdge(he,edgeIndex)
        return he


    def __bindVert(self, v1, v2):
        he=self.__genHalfEdge()
        self.__heAddVert(he, v1,v2)
        return he

    def __genHalfEdge(self):
        return uuid.uuid4().hex

    def __addEdge(self, edge):
        self.__edges.append(edge)
        return len(self.__edges)-1
    
    def __getFace(self,index):
        return self.__faces[index]
    
    def __getEdge(self,index):
        return self.__edges[index]
    
    def __getEdgeByVerts(self,v1,v2):
        if v1 in self.__vertsDict:
            for he in self.__vertsDict[v1]:
                info=self.__getHalfEdgeInfo(he)
                if info['vert_to']==v2:
                    return info['edge']
        if v2 in self.__vertsDict:
            for he in self.__vertsDict[v2]:
                info=self.__getHalfEdgeInfo(he)
                if info['vert_to']==v1:
                    return info['edge']
        raise Exception(f"Edge Not Found: [{v1},{v2}]")

    def __getOrAddEdge(self, v1, v2):
        if v1 in self.__vertsDict:
            for he in self.__vertsDict[v1]:
                info=self.__getHalfEdgeInfo(he)
                if info['vert_to']==v2:
                    return info['edge']
        if v2 in self.__vertsDict:
            for he in self.__vertsDict[v2]:
                info=self.__getHalfEdgeInfo(he)
                if info['vert_to']==v1:
                    return info['edge']
        return self.__addEdge([v1,v2])
    
    def __heAddVert(self, halfEdge, fromV, toV):
        self.__getHalfEdgeInfo(halfEdge)['vert_from'] = fromV
        self.__getHalfEdgeInfo(halfEdge)['vert_to'] = toV
        ls = self.__getListFromDict(self.__vertsDict,fromV)
        ls.append(halfEdge)

    def __heAddEdge(self, halfEdge, edgeIndex):
        self.__getHalfEdgeInfo(halfEdge)['edge']=edgeIndex
        ls = self.__getListFromDict(self.__edgeDict,edgeIndex)
        ls.append(halfEdge)
        if len(ls)>1:
            self.__heFlip(ls[0],ls[1])

    def __heAddFace(self, halfEdge, faceIndex):
        self.__getHalfEdgeInfo(halfEdge)['face']=faceIndex
        ls = self.__getListFromDict(self.__faceDict,faceIndex)
        ls.append(halfEdge)

    def __heFromTo(self, fromHalfEdge, toHalfEdge):
        self.__halfEdges[fromHalfEdge]['next'] = toHalfEdge
        self.__halfEdges[toHalfEdge]['pre'] = fromHalfEdge

    def __heFlip(self, he1, he2):
        self.__getHalfEdgeInfo(he1)['flip'] = he2
        self.__getHalfEdgeInfo(he2)['flip'] = he1

    def __getListFromDict(self, dict, key):
        if not key in dict:
            ls=[]
            dict[key]=ls
        return dict[key]
    
    def __getHalfEdgeInfo(self, halfEdge):
        if halfEdge not in self.__halfEdges:
            self.__halfEdges[halfEdge]={}
        return self.__halfEdges[halfEdge]
    
