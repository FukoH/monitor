class Relation:
    @property
    def pk_id(self):
        return self.__pk_id

    @property
    def op_time(self):
        return self.__op_time

    @property
    def index_pk_id(self):
        return self.__index_pk_id

    @property
    def parent_pk_id(self):
        return self.__parent_pk_id

    @property
    def distance(self):
        return self.__distance

    @property
    def is_leaf(self):
        return self.__is_leaf

    @property
    def influence_factor(self):
        return self.__influence_factor

    @pk_id.setter
    def pk_id(self, pk_id):
        self.__pk_id = pk_id

    @op_time.setter
    def op_time(self, op_time):
        self.__op_time = op_time

    @index_pk_id.setter
    def index_pk_id(self, index_pk_id):
        self.__index_pk_id = index_pk_id

    @parent_pk_id.setter
    def parent_pk_id(self, parent_pk_id):
        self.__parent_pk_id = parent_pk_id

    @distance.setter
    def distance(self, distance):
        self.__distance = distance

    @is_leaf.setter
    def is_leaf(self, is_leaf):
        self.__is_leaf = is_leaf

    @influence_factor.setter
    def influence_factor(self, influence_factor):
        self.__influence_factor = influence_factor
