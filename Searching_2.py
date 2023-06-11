import random
import math
class Node():
    def __init__(self, data, level, fval, father): #data-world state, level=g, fval=f-value(calculated)
        self.data=data
        self.level=level
        self.fval=fval
        self.father=father #keep the previous world state
        self.HueristicVal=None #keep the Heuristic val of every node
        self.rowInitial=None #keep for printing Simulated annealing
        self.columnInitial=None #keep for printing Simulated annealing
        self.rowFirstMove=None #keep for printing Simulated annealing
        self.columnFirstMove=None #keep for printing Simulated annealing
        self.FitnessProb = None #keep for the Genetic algo
        self.fromProb=None#keep for the Genetic algo
        self.untilProb=None#keep for the Genetic algo

    def generate_children(self): #generate child matrix (state) by check the all possible options
        children = []  # keep the children
        kingsList=self.find(self.data, 2)#find the kings location on the board
        bishopList=self.find(self.data, 3)#find the bishops location on the board
        children = self.findKingNeighbors(kingsList,children)
        children = self.findBishopsNeighbors(bishopList,children)
        return children

    def findBishopsNeighbors (self, bishopsList, children):
        for k in bishopsList:
            val_list = []  # bishop neighbors value positions
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
            x, y = k
            for i in val_list:
                child=self.shuffle(self.data,x,y,i[0],i[1])#switch with current bishop
                if child is not None:#None if is located outside the matrix
                        child_node = Node(child,self.level+1,0,self)#create child "World state"
                        children.append(child_node)#add to the children list
        return children

    def findKingNeighbors (self, kingsList, children):
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
        self.FM = []  # help list keep the indexes of the first moves - SA
        self.FirstBoards = []  # help list keep the first 3 boards for K-beams and Genetic
        self.flag = False  #help variable
        start = Node(self.starting_board, 0, 0, None)  # Initial "world state"
        goal = Node(self.goal_board, 0, 0, None)  # goal board
        if self.possibleGame(start, goal) == False:  # impossible goal board
            print("No path found")
            return
        match search_method:#switch case
            case 1:#A*
                self.run_A_star(start,goal)
            case 2:
                self.run_hillClimbing(start,goal)
            case 3:
                self.run_simulatedAnnealing(start,goal)
            case 4:
                self.run_K_Beams (start,goal)
            case 5:
                self.run_Genetic(start,goal)

    def heuristicCalculation(self,start,goal,S):#Calculate Heuristic value - +1 for every value that is not in his location
        temp = 0 #help variable
        for i in range (0,len(self.starting_board)):
            for j in range (0,len(self.starting_board)):
                if (start[i][j] == 2 or start[i][j]== 3): #King and Bishops location
                    if (start[i][j] != goal[i][j]):
                        temp+=1 #update
        S.HueristicVal=temp#keep the Heuristic value
        return temp

    def possibleGame (self,start,goal):#check if it's possible to reach the goal board
        if self.validBoard(start.data)==False or self.validBoard(goal.data)==False:#un valid board - player surrounded by forcefields
            print("No patch found")
            return False
        if self.unreachableBoard(start.data,goal.data)==False:##check if the bishops in the goal are compatible diagonals with the bishops in the start board
            print("No patch found")
            return False
        return True #possible goal board

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

#A*
    def run_A_star (self,start,goal): #run the Algo
        start.fval = self.f(start, goal)  # Calculate f
        self.open.append(start)  # put the start world state in the open list
        while True:#run until the ans founded
            if (bool(self.open)==False):#empty - no path found
                print("No path found")
                break
            cur = self.open[0]#take the smallest f_val
            if(self.heuristicCalculation(cur.data,goal.data,cur)==0): #found the ans
                self.printBoardsCourse(cur,goal)
                break
            self.checkNeighbors(cur, goal)#keep searching
            self.open.sort(key = lambda x:x.fval,reverse=False)#sort the open list bsed of f value

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

    def f(self,start,goal): #function to Calculate f=g+H
        return self.heuristicCalculation(start.data,goal.data,start) + start.level# calculate the Heuristic + g(Level)

#Hill climbing
    def run_hillClimbing (self,start,goal): #run Hill climbing
        current = Node(self.starting_board, 0, 0, None)  # same as start for keeping starting board clean
        if self.firstTryHC(current,goal): #finish at the first try
            return #won
        else:
            visitedChildren = []  # list of the children that we visited them already for no duplicate
            counterAttempts = 2  # count the attempts in case of failing - until 5
            while counterAttempts<=5:
                current, visitedChildren = self.HC_randomChoice(start, goal, visitedChildren)#random board
                if self.HC_Attempt(current,goal):#attempt check
                    return #won
                counterAttempts +=1 #another try
            print("No path found")
            return#lost

    def firstTryHC(self,current,goal):#boolean - first try hill climbing
        if(self.winCheck (current,goal)):
            return True #starting board = goal board
        if self.HC_Attempt(current,goal):
            return True #won
        return False#need 2nd try

    def HC_Attempt(self,current,goal):#attempt to reach the goal board
        current.HueristicVal = self.heuristicCalculation(current.data, goal.data, current)  # cal value
        while True:
            children = current.generate_children()  # starting board neighbors
            for i in children:  # calculate value for the children
                i.HueristicVal = self.heuristicCalculation(i.data, goal.data, i)  # calculate value for the child
            children.sort(key=lambda x: x.HueristicVal, reverse=False)  # sort the children list based of value
            if (current.HueristicVal > children[0].HueristicVal):  # better option
                current = children[0]  # update
                if (self.winCheck(current, goal)):
                    return True  # game over - won the game
            else:  # stuck
                return False  # need 2nd try

    def HC_randomChoice(self,start,goal,visitedChildren):
        children = start.generate_children()  # starting board neighbors
        for i in children:  # calculate value for the children
            i.HueristicVal = self.heuristicCalculation(i.data, goal.data, i)  # calculate value for the child
        children.sort(key=lambda x: x.HueristicVal, reverse=False)  # sort the children list based of value
        children.pop(0)  # remove the best child
        current = random.choice(children)  # choose randomly
        while True:
            for i in visitedChildren:
                if self.checkSameBoards(i.data, current.data):#identical
                    current = random.choice(children)  # choose randomly again
                    break
            visitedChildren.append(current)
            return current,visitedChildren #return values

#Simulated annealing
    def run_simulatedAnnealing(self,start, goal):
        if self.SA_Attempt(start,goal)==False:
            print("No path found")
            #return

    def SA_Attempt(self,current,goal):#simulated annealing attemp
        flag = True  # help variable - True when we made a move
        firstMove = True # is it the first move?
        t = 0  # count the moves
        current.HueristicVal = self.heuristicCalculation(current.data, goal.data, current)  # cal value
        while t <= 100:  # total 100 moves
            if flag == True: #made a move
                if self.winCheck(current, goal):  # won the game
                    return True
                T = self.calTemperature(t)  # value
                children = current.generate_children()  # board neighbors
                for i in children:  # calculate value for the children
                    i.HueristicVal = self.heuristicCalculation(i.data, goal.data, i)  # calculate value for the child
            deltaE = 0
            while deltaE==0: #same score
                nextBoard = random.choice(children)  # choose randomly
                deltaE = current.HueristicVal - nextBoard.HueristicVal  # Delta E
            if firstMove == True:
                self.copyFirstMove(nextBoard, current)
            if deltaE>0: #a better board
                if firstMove == True:
                    self.FM.append(1)#prob
                    firstMove = False
                current = nextBoard #update
                t+=1 #update
                flag=True #update
                continue #update
            prob = math.exp(deltaE/T)#keep the prob for change
            if firstMove == True:
                self.FM.append(prob)
            if random.random()<prob: #under the prob - rand (0,1)
                if firstMove == True:
                    firstMove = False
                current = nextBoard  # update
                t += 1  # update
                flag = True  # update
                continue  # update
            flag = False #another random
        return False # no path found

    def calTemperature (self,t): #calculate Temperature
        return (-0.09*t+10)

    def copyFirstMove(self,nextBoard,initialBoard): #check what is the first move and keep the indexes
        inBoard = initialBoard.data #keep the board
        neBoard = nextBoard.data
        for i in range(0,len(inBoard)):
            for j in range(0,len(inBoard)):
                if inBoard[i][j]!=neBoard[i][j] and neBoard[i][j] == 0:#previous place
                    initialBoard.rowInitial = i+1
                    initialBoard.columnInitial = j+1
                if inBoard[i][j]!=neBoard[i][j] and inBoard[i][j] == 0:#first move
                    initialBoard.rowFirstMove = i + 1
                    initialBoard.columnFirstMove = j + 1
        self.FM.append(initialBoard.rowInitial)
        self.FM.append(initialBoard.columnInitial)
        self.FM.append(initialBoard.rowFirstMove)
        self.FM.append(initialBoard.columnFirstMove)

#K - beams
    def run_K_Beams (self,current,goal):
        if self.winCheck(current, goal): #win check - start board = goal board
            return
        self.heuristicCalculation(current.data,goal.data,current) #cal start value
        children = current.generate_children()  # starting board neighbors
        firstMove = True #indicate if it is the first move
        while (True):
                for i in children:  # calculate value for the children
                    self.heuristicCalculation(i.data, goal.data, i)  # calculate value for the child
                children.sort(key=lambda x: x.HueristicVal, reverse=False)  # sort the children list based of value
                if self.winCheck(children[0], goal):#win check best board
                    return #won
                temp2 = []  # help list for saving children
                for j in range (0,3): #k-beams k=3
                    temp1 = children[j].generate_children()  #board neighbors
                    if firstMove == True:#save only the first 3 boards
                        self.FirstBoards.append(children[j])#add board
                    temp2 = self.copyChildren(temp1,temp2) #append children to temp2
                firstMove = False #update
                children=temp2 # update last beams

    def copyChildren (self,initial,end): #copy boards from one list to another
        for i in initial:
            end.append(i)
        return end #ans

#Genetic algo
    def run_Genetic(self,start,goal): #Genetic algorithm
        if self.winCheck(start, goal): #win check - start board = goal board
            return
        children = start.generate_children()  # starting board neighbors
        population = self.choosePopultion(children)
        self.Genetic_Attempt(population, goal)

    def Genetic_Attempt (self,initial_population,goal):
        curPopulation = initial_population #current population
        self.flag = True
        while True:#until found ans
            for i in curPopulation:  # calculate value for the initial population
                self.heuristicCalculation(i.data, goal.data, i)  # calculate value for the child
            if self.winFunction(curPopulation,goal): #win check
                return True
            self.FitnessFunction(curPopulation)  # Fitness function
            nextPopulation =[] #next population
            parents =[] #help variable
            while len(nextPopulation)<10:
                while(len(parents)<2): #2 parents
                    rand = random.randint(1, 100) #prob
                    for i in  initial_population:
                        if rand>=i.fromProb and rand<= i.untilProb:#prob for child
                            if i not in parents:#different parents
                                parents.append(i) #add parent
                                break
                nextPopulation = self.createChild(parents,nextPopulation)
                parents = []  # new parents
            curPopulation = nextPopulation #update

    def winFunction(self,population,goal): #check if one of the population is the goal board - win
        for i in population:
            if self.winCheck(i, goal):  # win check
                return True
        return False

    def createChild (self,parents,population): #create a board from 2 parents
        p1 = parents[0]
        p2 = parents[1]
        rand = random.randint(1,5)
        child =[] #child
        for i in range (0,rand):
            child.append(p1.data[i])
        for j in range (rand,len(p2.data)):
            child.append(p2.data[j])
        population =self.validChild(p1,p2,child,population)
        return population

    def validChild (self,p1,p2,child,population): #check if the child is valid
        if self.checkSameBoards(p1.data,child):#child == parent1
            if self.flag == True:
                self.FirstBoards.append(p1)
                self.FirstBoards.append(p1.FitnessProb)
                self.FirstBoards.append(p2)
                self.FirstBoards.append(p2.FitnessProb)
            if self.createMutation(p1,population):
                if self.flag == True:
                    self.FirstBoards.append(population[len(population)-1])#add last object
                    self.FirstBoards.append('yes')
                    self.flag=False
                return population#create mutation
            if self.flag == True:
                self.FirstBoards.append(p1)
                self.FirstBoards.append('no') #no mutation
                self.flag=False
            population.append(p1)
            return population
        if self.checkSameBoards(p2.data,child):#child == parent2
            if self.flag == True:
                self.FirstBoards.append(p1)
                self.FirstBoards.append(p1.FitnessProb)
                self.FirstBoards.append(p2)
                self.FirstBoards.append(p2.FitnessProb)
            if self.createMutation(p2,population):
                if self.flag == True:
                    self.FirstBoards.append(population[len(population) - 1])  # add last object
                    self.FirstBoards.append('yes')
                    self.flag = False
                return population#create mutation
            if self.flag == True:
                self.FirstBoards.append(p2)
                self.FirstBoards.append('no')  # no mutation
                self.flag = False
            population.append(p2)
            return population
        p2children = p2.generate_children()  # parent 2 children list
        for j in p2children:
            if self.checkSameBoards(j.data, child):  # child can be a child of this parent
                if self.flag == True:
                    self.FirstBoards.append(p1)
                    self.FirstBoards.append(p1.FitnessProb)
                    self.FirstBoards.append(p2)
                    self.FirstBoards.append(p2.FitnessProb)
                if self.createMutation(j, population):
                    if self.flag == True:
                        self.FirstBoards.append(population[len(population) - 1])  # add last object
                        self.FirstBoards.append('yes')
                        self.flag = False
                    return population  # create mutation
                if self.flag == True:
                    self.FirstBoards.append(j)
                    self.FirstBoards.append('no') #no mutation
                    self.flag = False
                population.append(j)
                return population
        p1children = p1.generate_children() #parent 1 children list
        for i in p1children:
            if self.checkSameBoards(i.data,child):#child can be a child of this parent
                if self.flag == True:
                    self.FirstBoards.append(p1)
                    self.FirstBoards.append(p1.FitnessProb)
                    self.FirstBoards.append(p2)
                    self.FirstBoards.append(p2.FitnessProb)
                if self.createMutation(p2, population):
                    if self.flag == True:
                        self.FirstBoards.append(population[len(population) - 1])  # add last object
                        self.FirstBoards.append('yes')
                        self.flag = False
                    return population  # create mutation
                if self.flag == True:
                    self.FirstBoards.append(i)
                    self.FirstBoards.append('no') #no mutation
                    self.flag = False
                population.append(i)
                return population
        return population#no child added

    def createMutation (self,child,population):#create mutation
        r = random.randint(0,100)
        if r<=20: #prob for mutation
            children = child.generate_children()
            Mutation = random.choice(children)  # choose randomly
            population.append(Mutation)
        return population

    def choosePopultion (self,children): #choose initial population from starting board
        ans = [] #answer
        for i in range (0,10): #Popultion = 10
            nextBoard = random.choice(children)  # choose randomly
            children.remove(nextBoard)
            ans.append(nextBoard)
        return ans

    def FitnessFunction (self,population): #cal probabilities to choose each board
        sum = 0 #sum of population values
        temp=0 #help variable
        for i in population:
            sum += 6-(i.HueristicVal) # 6=max value - worst
        for j in population:
            j.FitnessProb = ((6-(j.HueristicVal))/sum)*100#precentage - give a better prob for the better boards
            j.fromProb = temp #for us to know in which prob to choose
            untilValue = temp+j.FitnessProb
            j.untilProb = untilValue #for us to know in which prob to choose
            temp = untilValue #update
#
    
    def winCheck (self,current,goal):#win check
        if (self.heuristicCalculation(current.data, goal.data,current) == 0):  # found the ans
            self.printBoardsCourse(current, goal)
            return True  # end program
        return False#else

    def printBoardsCourse(self, current, goal):  # print the board that lead to the goal board
        temp = []  # board from the last one to the first one
        while current != None:  # reach to the Initial board
            temp.append(current)  # organize the boards
            current = current.father  # update
        boardsList = self.reverseOrder(temp)
        if self.detail_output == False:  # regular printing
            self.regularPrinting(boardsList)
            return
        match self.search_method:  # switch case
            case 1:
                self.AstarSpecialPrint(boardsList)  # A* special print
            case 2:
                self.regularPrinting(boardsList)  # Hill climbing no change
            case 3:
                self.SAspecialPrint(boardsList)  # Simulated annealing special print
            case 4:
                self.KbeamsSpecialPrint(boardsList)  # K-beams special print
            case 5:
                self.GeneticSpecialPrint(boardsList) #Genetic special print

    def AstarSpecialPrint(self, boardsList):  # A* special print
        print("Board 1 (starting position):")  # first board
        self.Print(boardsList[0].data)
        print('_____')
        boardNum = 2
        i = 1
        secondBoard = True
        while (i < len(boardsList) - 1):  # print boards until goal board
            if secondBoard == True:  # second board
                print("Board " + str(boardNum) + ":")
                self.Print(boardsList[i].data)
                i += 1  # update
                boardNum += 1
                print("Heuristic: " + str(boardsList[i].HueristicVal))  # print the Heuristic value
                print('_____')
                secondBoard = False
                continue
            print("Board " + str(boardNum) + ":")
            self.Print(boardsList[i].data)
            i += 1  # update
            boardNum += 1
            print('_____')
        print("Board " + str(boardNum) + " (goal position):")  # goal board
        self.Print(boardsList[i].data)  # goal board
        print('_____')

    def SAspecialPrint(self, boardsList):  # Simulated annealing special print
        print("Board 1 (starting position):")  # first board
        self.Print(boardsList[0].data)
        i = 0  # help variable
        while (i < len(self.FM)):
            print('action:(' + str(self.FM[i]) + ',' + str(self.FM[i + 1]) + ')->(' + str(self.FM[i + 2]) + ',' + str(
                self.FM[i + 3]) + '); probability: ' + str(self.FM[i + 4]))
            i = i + 5  # update
        print('_____')
        boardNum = 2
        i = 1
        while (i < len(boardsList) - 1):  # print boards until goal board
            print("Board " + str(boardNum) + ":")
            self.Print(boardsList[i].data)
            i += 1  # update
            boardNum += 1
            print('_____')
        print("Board " + str(boardNum) + " (goal position):")  # goal board
        self.Print(boardsList[i].data)  # goal board
        print('_____')

    def KbeamsSpecialPrint(self, boardsList):
        print("Board 1 (starting position):")  # first board
        self.Print(boardsList[0].data)
        print('_____')
        i = 0  # help variable
        while (i < len(self.FirstBoards)):
            if i == 0:
                print("Board 2a:")
                self.Print(self.FirstBoards[i].data)
                print('_____')
            elif i == 1:
                print("Board 2b:")
                self.Print(self.FirstBoards[i].data)
                print('_____')
            else:  # i==2
                print("Board 2c:")
                self.Print(self.FirstBoards[i].data)
                print('_____')
            i += 1  # update
        boardNum = 2
        i = 1
        while (i < len(boardsList) - 1):  # print boards until goal board
            print("Board " + str(boardNum) + ":")
            self.Print(boardsList[i].data)
            i += 1  # update
            boardNum += 1
            print('_____')
        print("Board " + str(boardNum) + " (goal position):")  # goal board
        self.Print(boardsList[i].data)  # goal board
        print('_____')

    def GeneticSpecialPrint(self, boardsList):
        self.regularPrinting(boardsList)
        print('starting board 1 (probability of selection from population::<'+str(self.FirstBoards[1])+'>):')
        self.Print(self.FirstBoards[0].data)
        print('starting board 2 (probability of selection from population::<' + str(self.FirstBoards[3])+'>):')
        self.Print(self.FirstBoards[2].data)
        print('Result board (mutation happend::<'+str(self.FirstBoards[5])+'>):')
        self.Print(self.FirstBoards[4].data)

    def reverseOrder(self, temp):  # reverse order list
        i = len(temp) - 1
        ans = []
        while (i >= 0):
            ans.append(temp[i])  # add board
            i -= 1  # update
        return ans

    def regularPrinting(self, boardsList):  # detail_output == False
        print("Board 1 (starting position):")  # first board
        self.Print(boardsList[0].data)
        print('_____')
        boardNum = 2
        i = 1
        while (i < len(boardsList) - 1):  # print boards until goal board
            print("Board " + str(boardNum) + ":")
            self.Print(boardsList[i].data)
            i += 1  # update
            boardNum += 1
            print('_____')
        print("Board " + str(boardNum) + " (goal position):")  # goal board
        self.Print(boardsList[i].data)  # goal board
        print('_____')

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
#END
#####------------------------------------------------#####
class Static:
    @staticmethod
    def find_path(starting_board, goal_board, search_method, detail_output):  # given function
        G = Game(starting_board, goal_board, search_method, detail_output)  # start the game


