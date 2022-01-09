"""Containers module."""

from dependency_injector import containers
from detector import Detector


class Container(containers.DeclarativeContainer):
    
    detector = Detector()