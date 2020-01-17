def make_coding_function2D(X, coding_length):
    def chunk(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    async def coding_function(self,loop,i,j):
        X_sum = None
        coding_length2 = int(len(X._block_idxs(0))/coding_length)
        coding_chunks = list(chunk(sorted(X._block_idxs(0)), coding_length2))
        if (i <= max(X._block_idxs(0))):
            return X.get_block(i,j)
        elif (i < len(X._block_idxs(0)) + coding_length2):
            left = i - len(X._block_idxs(0))
            for c in range(coding_length2):
                t = c*coding_length2 + left
                if (X_sum is None):
                    X_sum = X.get_block(t,j)
                else:
                    X_sum = X_sum + X.get_block(t,j) 
            self.put_block(X_sum, i, j)   
            return X_sum
        elif (i < len(X._block_idxs(0)) + coding_length2 + coding_length):
            left = i - len(X._block_idxs(0)) - coding_length2
            for c in coding_chunks[left]:
                if (X_sum is None):
                    X_sum = X.get_block(c,j)
                else:
                    X_sum = X_sum + X.get_block(c,j) 
            self.put_block(X_sum, i, j)
            return X_sum
        elif (i == len(X._block_idxs(0)) + coding_length2 + coding_length):
            for c in range(coding_length2):
                t = c + len(X._block_idxs(0))
                if (X_sum is None):
                    X_sum = self.get_block(t,j)
                else:
                    X_sum = X_sum + self.get_block(t,j) 
            self.put_block(X_sum, i, j)   
            return X_sum
        else:
            print ("ERROR: Encoding something not already signified")
            return None
    return coding_function