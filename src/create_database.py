from models import data_access
from models.models import Base

# data_access.engine.create_all()
# db.session.commit()
Base.metadata.create_all(data_access.engine)

from models.featureset import Base

# data_access.engine.create_all()
# db.session.commit()
Base.metadata.create_all(data_access.engine)
