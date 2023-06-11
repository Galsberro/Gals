
class Node():
    def __init__(self, data, level, fval, father): #data-world state, level=g, fval=f-value(calculated)
        self.data=data
        self.level=level
        self.fval=fval
        self.father=father #keep the previous world state
        self.HueristicVal=0 #keep the Heuristic val of every node

    def generate_children(self): #generate child matrix (state) by check the all possible options
        children = []  # keep the children
        kingsList=self.find(self.data, 2)#find the kings location on the board
        bishopList=self.find(self.data, 3)#find the bishops location on the board
        children = self.findKingNeighbors(kingsList,children)
        children = self.findBishopsNeighbors(bishopList,children)
        return children

    def findBishopsNeighbors (self, bishopsList, children):
        val_list = []  # bishop neighbors value positions
        for k in bishopsList:
            x,y = k
            while y < len(self.data)-1 and x > 0:#upper-right diagonal
                if self.data[x-1][y+1]!=0:#occupied place
                    break
                val_list.append([x-1,y+1])
                x-=1
                y+=1
            x,y = k
            while y>0 and x>0:  # upper-left diagonal
                    if self.data[x-1][y-1]!= 0:  # occupied place
                        break
                    val_list.append([x-1,y-1])
                    x-=1
                    y-=1
            x,y = k
            while y < len(self.data)-1 and x < len(self.data)-1 :  # right-down diagonal
                    if self.data[x+1][y+1]!= 0:  # occupied place
                        break
                    val_list.append([x+1,y+1])
                    x+=1
                    y+=1
            x,y = k
            while y >0 and x < len(self.data)-1 :  # left-down diagonal
                    if self.data[x+1][y-1]!= 0:  # occupied place
                        break
                    val_list.append([x+1,y-1])
                    x+=1
                    y-=1
        for i in val_list:
            child=self.shuffle(self.data,x,y,i[0],i[1])#switch with current bishop
            if child is not None:#None if is located outside the matrix
                    child_node = Node(child,self.level+1,0,self)#create child "World state"
                    children.append(child_node)#add to the children list
        return children

    def  findKingNeighbors (self, kingsList, children):
        for k in kingsList:
            x,y = k
            val_list = [[x,y-1],[x,y+1],[x-1,y],[x+1,y],[x+1,y+1],[x+1,y-1],[x-1,y-1],[x-1,y+1]]#king neighbors value positions -8
            for i in val_list:
                child=self.shuffle(self.data,x,y,i[0],i[1])#switch with current king
                if child is not None:#None if is located outside the matrix
                    child_node = Node(child,self.level+1,0,self)#create child "World state"
                    children.append(child_node)#add to the children list
            val_list=[]#reset for no double boards
        return children

    def shuffle(self, puz, x1, y1, x2, y2): #puz = the current matrix
        #Move the king in the given direction and if the position value are out of limits the return None
        if x2 >= 0 and x2 < len(self.data) and y2 >= 0 and y2 < len(self.data):#in the matrix's boundaries
            if (self.data[x2][y2]==1 or self.data[x2][y2]==2 or self.data[x2][y2]==3):#cannot swith - already manned
                return None
            temp_puz = []
            temp_puz = self.copy(puz) #copy the board
            temp = temp_puz[x2][y2] #help variable
            temp_puz[x2][y2] = temp_puz[x1][y1]
            temp_puz[x1][y1] = temp
            return temp_puz #return the board (data)
        else:
            return None #out of boundaries

    def copy(self, root): #root = board that we want to be copied,Copy function to create a similar matrix of the given node
        temp = []
        for i in root:
            t = []
            for j in i:
                t.append(j)
            temp.append(t)
        return temp  #return the board

    def find(self, puz, x): #finding the kings and bishops
        ans = [] #keep the wanted location
        for i in range (0,len(self.data)):
            for j in range (0,len(self.data)):
                if puz [i][j] == x: # found the king / bishop
                   ans.append([i,j])#add to wanted location list
        return ans

class Game():
    def __init__ (self,starting_board, goal_board, search_method, detail_output):
        self.open = [] #frontier , neighbors - "world states" that we haven't visit yet
        self.closed = [] #neighbors - "world states" that we visited
        self.starting_board = starting_board
        self.goal_board = goal_board
        self.detail_output=detail_output
        self.search_method=search_method#for now doing nothing
        self.run_A_star()

    def run_A_star (self): #run the Algo
        start = Node(self.starting_board, 0, 0, None)  # Initial "world state"
        goal = Node(self.goal_board, 0, 0, None)  # goal board
        start.fval = self.f(start, goal)  # Calculate f
        self.open.append(start)  # put the start world state in the open list
        if self.validBoard(start.data)==False or self.validBoard(goal.data)==False:#un valid board - player surrounded by forcefields
            print("No patch found")
            return
        if self.unreachableBoard(start.data,goal.data)==False:##check if the bishops in the goal are compatible diagonals with the bishops in the start board
            print("No patch found")
            return
        while True:#run until the ans founded
            if (bool(self.open)==False):#empty - no path found
                print("No patch found")
                break
            cur = self.open[0]#take the smallest f_val
            if(self.heuristicCalculation(cur.data,goal.data,cur)==0): #found the ans
                self.printBoardsCourse(cur,goal)
                break
            self.checkNeighbors(cur, goal)#keep searching
            self.open.sort(key = lambda x:x.fval,reverse=False)#sort the open list bsed of f value

    def unreachableBoard (self,start,goal): #check if the bishops in the goal are compatible diagonals with the bishops in the start board
        GoalEvenDiagonalSet,GoalOddDiagonalSet=self.checkLocations(goal)#check goal board
        StartEvenDiagonalSet,StartOddDiagonalSet=self.checkLocations(start)#check start board
        if GoalEvenDiagonalSet!=StartEvenDiagonalSet or GoalOddDiagonalSet!=StartOddDiagonalSet:#
            return False #impossible game
        return True # possible game

    def checkLocations (self, Board):#check the bishops location - Even / Odd
        EvenDiagonalSet = 0
        OddDiagonalSet = 0
        for i in range(0, len(Board)):
            for j in range(0, len(Board)):
                if Board[i][j] == 3:  # bishop location
                    if (i + j) % 2 == 0:  # Even diagonal location
                        EvenDiagonalSet += 1
                    else: # Odd diagonal location
                        OddDiagonalSet += 1
        return EvenDiagonalSet,OddDiagonalSet

    def validBoard (self,Board):# check if king is surrounded by forcefields by force fields
        flag = False # help variable
        kingNum = 0 # count the kings
        for i in range (0,len(Board)):
            if kingNum == 3:
                break
            for j in range (0,len(Board)):
                if kingNum == 3:
                    break
                flag = False
                if Board[i][j]==2: #king location
                    kingNum +=1
                    val_list = [[i,j-1],[i,j+1],[i-1,j],[i+1,j],[i+ 1,j+1],[i+1,j-1],
                                [i-1,j-1],[i-1,j+1]]#king neighbors
                    for k in val_list:
                        if k[0] >= 0 and k[0] < len(Board) and k[1] >= 0 and k[1] < len(Board):
                            if Board[k[0]][k[1]] != 1:
                                flag = True #un surrounded king
                                break
        if flag == False: #surrounded by forcefield
              return False#surrounded by forcefield

    def printBoardsCourse(self,cur,goal):#print the board that lead to the goal board
        current = cur
        temp =[]#help for printing
        H=0
        while current != None:# reach to the Initial board
            temp.append(current)#organize the boards
            if current.father == None:
                H=current.HueristicVal
            current = current.father #update
        o=len(temp)-1
        print("Board 1 (starting position):")#first board
        self.Print(temp[o].data)
        print('_____')
        counter = 2
        while (o!=1):#print from the end to begin exept goal board
            print("Board " + str(counter) + ":")
            o -=1 #update
            self.Print(temp[o].data)
            counter +=1
            if self.detail_output == True and o==len(temp)-2:
                print("Heuristic: "+str(H))#print the Heuristic value
            print('_____')
        print("Board "+str(counter)+" (goal position):")#goal board
        self.Print(goal.data)#goal board

    def AlreadyOnTheLists (self,board1):#check if the know about the world-state already for not appending him to the lists
        for i in range (0,len(self.open)):#check all the open list for same board
            if self.checkSameBoards (self.open[i].data,board1):#found same board
                return True #the item will not be added to the list
        for j in range (0,len(self.closed)):#check all the closed list for same board
            if self.checkSameBoards(self.closed[j].data, board1):  # found same board
                return True
        return False # new world state, need to be added

    def checkSameBoards (self,A,B): #check if 2 board the same - A,B
        for i in range(len(A)):
            for j in range(0,len(A)):
                if A[i][j]!=B[i][j]:#found something different
                    return False#different boards
        return True#same boards

    def checkNeighbors (self, current, goal):
        for i in current.generate_children(): #return all the possible neighbors "world-state"
            if self.AlreadyOnTheLists(i.data):#already appear on the lists
                continue
            i.fval = self.f(i,goal)#for each calculate the f_val
            self.open.append(i)#add to the neighbors that have'nt visited yet
        self.closed.append(current) #add the visited to the closed list
        del self.open[0]#remove from the not-visited list

    def Print(self,B):  # Print according to the instruction
        print("  1 2 3 4 5 6")
        for i in range(0, len(B)):
            print(i + 1, end=':')
            for j in range(0, len(B)):
                if (B[i][j] == 0):
                    print(' ', end=' ')
                if (B[i][j] == 1):
                    print('@', end=' ')
                if (B[i][j] == 2):
                    print('*', end=' ')
                if (B[i][j] == 3):
                    print('&', end=' ')
            print()

    def f(self,start,goal): #function to Calculate f=g+H
        return self.heuristicCalculation(start.data,goal.data,start) + start.level# calculate the Heuristic + g(Level)

    def heuristicCalculation(self,start,goal,S):#Calculate Heuristic value - +1 for every value that is not in his location
        temp = 0 #help variable
        for i in range (0,len(self.starting_board)):
            for j in range (0,len(self.starting_board)):
                if (start[i][j] == 2 or start[i][j]== 3): #King and Bishops location
                    if (start[i][j] != goal[i][j]):
                        temp+=1 #update
        S.HueristicVal=temp#keep the Heuristic value
        return temp

#END
#####------------------------------------------------#####
class Static:
    @staticmethod
    def find_path(starting_board, goal_board, search_method, detail_output):  # given function
        G = Game(starting_board, goal_board, search_method, detail_output)  # start the game
